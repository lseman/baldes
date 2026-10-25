//! Automatic stabilization checkpoints contain original model geometry only.
use super::*;
use crate::master::stabilization::proximal::{ProximalParameters, ProximalSnapshot, ProximalState};

#[derive(Clone, Debug)]
pub struct ProximalWarmStart {
    pub snapshot: ProximalSnapshot,
    parameters: ProximalParameters,
    senses: Vec<RowSense>,
    rhs: Vec<f64>,
    roles: Vec<CurvatureRow>,
    offset: f64,
    columns: Vec<(usize, f64, Vec<(usize, f64)>)>,
    pub(crate) child: bool,
    pub(crate) center_invalid: bool,
}

pub(super) fn roles<M, C: Controller<M>>(
    controller: &C,
    count: usize,
) -> Result<Vec<CurvatureRow>> {
    let mut rows = controller.curvature_rows(count);
    if rows.len() != count {
        return Err(invalid("curvature row metadata length"));
    }
    for row in controller.unstabilized_rows() {
        let entry = rows
            .get_mut(row)
            .ok_or_else(|| invalid("curvature excluded row"))?;
        entry.class = CurvatureRowClass::Excluded;
    }
    Ok(rows)
}

impl ProximalWarmStart {
    pub(super) fn capture<M: LpMaster, C: Controller<M>>(
        state: &ProximalState,
        master: &M,
        controller: &C,
        parameters: &ProximalParameters,
        artificials: &[usize],
        cut_artificials: &[(usize, usize)],
    ) -> Result<Option<Self>> {
        if parameters.curvature.is_none() {
            return Ok(None);
        }
        let Some(snapshot) = state.snapshot() else {
            return Ok(None);
        };
        let mut columns = vec![];
        for (index, &cost) in master.costs().iter().enumerate() {
            if artificials.contains(&index) || cut_artificials.iter().any(|&(_, c)| c == index) {
                continue;
            }
            columns.push((index, cost, master.column_entries(index)?));
        }
        Ok(Some(Self {
            snapshot,
            parameters: parameters.clone(),
            senses: master.senses().to_vec(),
            rhs: master.rhs().to_vec(),
            roles: roles(controller, master.rhs().len())?,
            offset: master.objective_offset(),
            columns,
            child: false,
            center_invalid: false,
        }))
    }

    pub(super) fn restore<M: LpMaster, C: Controller<M>>(
        &self,
        master: &M,
        controller: &C,
        parameters: &ProximalParameters,
    ) -> Result<Option<ProximalSnapshot>> {
        if !same_parameters(&self.parameters, parameters) {
            return Ok(None);
        }
        let count = self.rhs.len();
        let mut compatible = !self.center_invalid
            && master.rhs().len() >= count
            && master.senses().get(..count) == Some(self.senses.as_slice())
            && roles(controller, master.rhs().len())?.get(..count) == Some(self.roles.as_slice())
            && (self.child
                || (master.rhs().get(..count) == Some(self.rhs.as_slice())
                    && master.objective_offset() == self.offset));
        if compatible {
            for (index, cost, entries) in &self.columns {
                if master.costs().get(*index) != Some(cost) {
                    compatible = false;
                    break;
                }
                let actual: Vec<_> = master
                    .column_entries(*index)?
                    .into_iter()
                    .filter(|&(r, _)| r < count)
                    .collect();
                if &actual != entries {
                    compatible = false;
                    break;
                }
            }
        }
        let mut snapshot = self.snapshot.clone();
        if compatible {
            snapshot.center.resize(master.rhs().len(), 0.);
        } else if self.child && snapshot.curvature.is_some() {
            // Child global radius information is dimension-independent. Never
            // assign old coordinates to different rows or carry a bound.
            snapshot.center = vec![0.; master.rhs().len()];
        } else {
            return Ok(None);
        }
        snapshot.bound = f64::NEG_INFINITY;
        Ok(Some(snapshot))
    }
}

fn same_parameters(a: &ProximalParameters, b: &ProximalParameters) -> bool {
    a.curvature == b.curvature
        && a.multi_point == b.multi_point
        && a.max_points == b.max_points
        && a.bands.len() == b.bands.len()
        && a.bands
            .iter()
            .zip(&b.bands)
            .all(|(a, b)| a.half_width == b.half_width && a.penalty == b.penalty)
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/cg/automatic_proximal_tests.rs"]
mod tests;
