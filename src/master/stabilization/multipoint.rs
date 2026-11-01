//! Support-plane epigraph for a multipoint proximal pricing model.
use crate::master::{LpMaster, RowSense};
use crate::{invalid, Result};

/// Support-plane retention metadata. Dual activity uses the positive activity
/// convention for <= rows (the negative of the generic minimization LP dual).
#[derive(Clone, Debug)]
pub struct SupportPlane {
    pub point: Vec<f64>,
    pub dual_activity: f64,
    pub zero_age: u32,
    pub protected_center: bool,
    pub creation_id: u64,
}

/// Record a completed private LP's support-row activity. Missing/invalid
/// observations leave all metadata unchanged.
pub fn observe_activity(planes: &mut [SupportPlane], activity: Option<&[f64]>) -> bool {
    let Some(activity) = activity else {
        return false;
    };
    if activity.len() != planes.len() || activity.iter().any(|d| !d.is_finite()) {
        return false;
    }
    for (plane, &dual) in planes.iter_mut().zip(activity) {
        plane.dual_activity = dual;
        if dual.abs() <= dual.abs() * 1e-6 + 1e-6 {
            plane.zero_age = plane.zero_age.saturating_add(1);
        } else {
            plane.zero_age = 0;
        }
    }
    true
}

/// Protected centers first, then descending dual activity, increasing zero
/// age, and decreasing creation ID. Buffered stable sorting preserves the
/// collection ordering even when tolerant comparisons are not transitive.
pub fn retain_planes(planes: &mut Vec<SupportPlane>, capacity: usize) -> Result<()> {
    if capacity == 0 || planes.iter().any(|p| !p.dual_activity.is_finite()) {
        return Err(invalid("invalid multipoint retention state"));
    }
    let mut order: Vec<_> = (0..planes.len()).collect();
    crate::master::ordering::stable(
        &mut order,
        &|a, b| precedes(&planes[a], &planes[b]),
        &mut crate::master::ordering::Work::new(None),
    )
    .ok_or_else(|| invalid("multipoint support-plane ordering interrupted"))?;
    let mut previous: Vec<_> = std::mem::take(planes).into_iter().map(Some).collect();
    planes.extend(
        order
            .into_iter()
            .map(|index| previous[index].take().unwrap()),
    );
    let keep = planes
        .iter()
        .take(capacity)
        .take_while(|p| p.protected_center || p.zero_age <= 2)
        .count();
    planes.truncate(keep);
    Ok(())
}

fn precedes(a: &SupportPlane, b: &SupportPlane) -> bool {
    if a.protected_center != b.protected_center {
        return a.protected_center;
    }
    let scale = a.dual_activity.abs().max(b.dual_activity.abs());
    // Native equality compares both subtraction orders. At large activities,
    // abs(a-b) can disagree at an adjacent binary64 boundary.
    let equality_margin = scale * 1e-6 + 1e-6;
    if a.dual_activity < b.dual_activity - equality_margin
        || b.dual_activity < a.dual_activity - equality_margin
    {
        return b.dual_activity < a.dual_activity - (scale * 1e-10 + 1e-6);
    }
    if a.zero_age != b.zero_age {
        return a.zero_age < b.zero_age;
    }
    a.creation_id > b.creation_id
}

/// Install `beta >= sample . (positive_outer - negative_outer)` planes.
/// Beta is free and has objective coefficient one; splitting it into two
/// nonnegative variables keeps the generic LP interface unchanged.
/// All columns/rows must belong to a private pricing master.
pub fn add_support_planes<M: LpMaster>(
    private: &mut M,
    outer_columns: &[(usize, usize, usize)],
    samples: &[Vec<f64>],
) -> Result<()> {
    add_support_plane_refs(private, outer_columns, samples.iter().map(Vec::as_slice))
}

pub(super) fn add_support_plane_refs<'a, M: LpMaster>(
    private: &mut M,
    outer_columns: &[(usize, usize, usize)],
    samples: impl Iterator<Item = &'a [f64]> + Clone,
) -> Result<()> {
    if samples.clone().next().is_none()
        || samples.clone().any(|s| {
            s.iter().any(|x| !x.is_finite()) || outer_columns.iter().any(|&(r, _, _)| r >= s.len())
        })
    {
        return Err(invalid("invalid multipoint support planes"));
    }
    let positive_beta = private.add_column(1., &[])?;
    let negative_beta = private.add_column(-1., &[])?;
    for sample in samples {
        let mut coefficients = vec![(positive_beta, -1.), (negative_beta, 1.)];
        for &(row, negative, positive) in outer_columns {
            coefficients.push((negative, -sample[row]));
            coefficients.push((positive, sample[row]));
        }
        private.add_row(RowSense::LessEqual, 0., &coefficients)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/stabilization/multipoint/tests.rs"]
mod tests;
