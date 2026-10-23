//! Independent shared-master variables with original bounds and coefficients.
use super::*;
use crate::master::preprocessing::Bounds;
mod projected;
pub(super) mod publish;
pub(super) use projected::infer as projected_bounds;

/// An original master variable, independent of route generation. Bounds may
/// be signed, one-sided or free. Infinite endpoints are mathematical domains,
/// not finite backend sentinels. Integer variables retain their
/// integrality in complete-domain MIP, while root masters remain continuous.
#[derive(Clone, Debug)]
pub struct MasterVariable {
    pub id: i32,
    pub cost: f64,
    pub bounds: Bounds,
    pub integer: bool,
    /// Coalesced coefficients on original shared rows.
    pub entries: Vec<(usize, f64)>,
}

pub(super) fn valid_bounds(bounds: Bounds) -> bool {
    !bounds.lower.is_nan()
        && !bounds.upper.is_nan()
        && bounds.lower != f64::INFINITY
        && bounds.upper != f64::NEG_INFINITY
        && bounds.lower <= bounds.upper
}

// Negative metadata tags keep static columns out of generated-route cleanup.
// -1 remains reserved for artificial variables. Both components are >=0:
// x = positive - negative. This preserves signed bounds without rounded shifts.
pub(super) fn decode(tag: i32) -> Option<(usize, f64)> {
    (tag <= -2).then(|| {
        let code = (-i64::from(tag) - 2) as usize;
        (code / 2, if code % 2 == 0 { 1. } else { -1. })
    })
}
pub(super) fn upper(variable: &MasterVariable, sign: f64) -> f64 {
    component_upper(variable.bounds, sign)
}
pub(super) fn component_upper(bounds: Bounds, sign: f64) -> f64 {
    if sign > 0. {
        bounds.upper.max(0.)
    } else {
        (-bounds.lower).max(0.)
    }
}
pub(super) fn coefficients<const D: usize>(
    state: &State<D>,
    tag: i32,
) -> Result<Vec<(usize, f64)>> {
    let (index, sign) = decode(tag).ok_or_else(|| invalid("master-variable metadata"))?;
    let variable = state
        .master_variables
        .get(index)
        .ok_or_else(|| invalid("master-variable index"))?;
    Ok(variable
        .entries
        .iter()
        .map(|&(r, a)| (state.shared_offset + r, sign * a))
        .collect())
}

pub(super) fn validate<const D: usize>(
    problem: &Problem<D>,
    variables: &[MasterVariable],
) -> Result<()> {
    let original_rows = problem.rows.len();
    let mut ids = HashSet::new();
    for variable in variables.iter() {
        if variable.id < 0
            || !ids.insert(variable.id)
            || !variable.cost.is_finite()
            || !valid_bounds(variable.bounds)
        {
            return Err(invalid("master-variable identity, cost or bounds"));
        }
        let mut rows = HashSet::new();
        if variable
            .entries
            .iter()
            .any(|&(r, a)| r >= original_rows || !a.is_finite() || !rows.insert(r))
        {
            return Err(invalid("coalesced master-variable coefficients required"));
        }
    }
    Ok(())
}

pub(super) fn install<const D: usize>(
    problem: &mut Problem<D>,
    variables: &mut [MasterVariable],
) -> Result<()> {
    validate(problem, variables)?;
    for variable in variables {
        for (sense, rhs) in [
            (RowSense::GreaterEqual, variable.bounds.lower),
            (RowSense::LessEqual, variable.bounds.upper),
        ] {
            if !rhs.is_finite() {
                continue;
            }
            let row = problem.rows.len();
            problem.rows.push(Row { sense, rhs });
            variable.entries.push((row, 1.));
        }
    }
    Ok(())
}

pub(super) fn load<const D: usize>(
    master: &mut ClpMaster,
    state: &State<D>,
) -> Result<Vec<cg::Column>> {
    let mut columns = vec![];
    for (index, variable) in state.master_variables.iter().enumerate() {
        for (side, sign) in [(0usize, 1.), (1, -1.)] {
            let tag = i32::try_from(
                2usize
                    .checked_mul(index)
                    .and_then(|v| v.checked_add(side + 2))
                    .ok_or_else(|| invalid("master-variable tag overflow"))?,
            )
            .map_err(|_| invalid("master-variable tag overflow"))?;
            let entries = coefficients(state, -tag)?;
            let column = master.add_column(sign * variable.cost, &entries)?;
            // The backend accepts a finite infinity sentinel. It must not
            // become a semantic finite bound or retain the default 1e12 cap.
            let limit = upper(variable, sign);
            master.set_upper(column, if limit.is_infinite() { f64::MAX } else { limit })?;
            columns.push(cg::Column {
                cost: sign * variable.cost,
                reduced_cost: 0.,
                entries,
                acceptable: true,
                tag: -tag,
                identity: format!("master:{}:{side}", variable.id),
            });
        }
    }
    Ok(columns)
}

pub(super) fn correction<const D: usize>(
    state: &State<D>,
    duals: &[f64],
) -> Option<(f64, Vec<(usize, f64)>)> {
    let mut correction = 0.;
    let mut gradient = vec![];
    let mut projected = None;
    for (index, variable) in state.master_variables.iter().enumerate() {
        for sign in [1., -1.] {
            let reduced = variable
                .entries
                .iter()
                .fold(sign * variable.cost, |c, &(r, a)| {
                    c - duals[state.shared_offset + r] * sign * a
                });
            if !reduced.is_finite() {
                return None;
            }
            let value = if reduced < 0. {
                let cap = upper(variable, sign);
                if cap.is_finite() {
                    cap
                } else {
                    if projected.is_none() {
                        projected = Some(projected_bounds(state)?);
                    }
                    component_upper(projected.as_ref().unwrap()[index], sign)
                }
            } else {
                0.
            };
            if !value.is_finite() {
                return None;
            }
            correction += reduced * value;
            if !correction.is_finite() {
                return None;
            }
            for &(r, a) in &variable.entries {
                let activity = value * sign * a;
                if !activity.is_finite() {
                    return None;
                }
                gradient.push((state.shared_offset + r, activity));
            }
        }
    }
    Some((correction, gradient))
}

impl<const D: usize> NodeSnapshot<D> {
    /// Restrict/fix an existing original master variable, preserving parent and
    /// sibling models. This adds original rows; it never widens the parent.
    pub fn with_master_variable_bounds(&self, id: i32, bounds: Bounds) -> Result<Self> {
        if !self.state.master_variables.iter().any(|v| v.id == id) {
            return Err(invalid("unknown master variable"));
        }
        if !valid_bounds(bounds) {
            return Err(invalid("ordered master-variable bounds required"));
        }
        let mut node = self.fork()?;
        for (sense, rhs) in [
            (RowSense::GreaterEqual, bounds.lower),
            (RowSense::LessEqual, bounds.upper),
        ] {
            if rhs.is_finite() {
                node = node.with_master_row(Row { sense, rhs }, &[], &[(id, 1.)])?;
            }
        }
        Ok(node)
    }
}

pub(super) fn add_product(sum: &mut crate::master::root::ExactSum, a: f64, b: f64) -> Option<()> {
    if b.fract() == 0. && b.abs() <= 9_007_199_254_740_992. {
        return sum.product(a, b);
    }
    if a.fract() == 0. && a.abs() <= 9_007_199_254_740_992. {
        return sum.product(b, a);
    }
    let product = a * b;
    if a == 0. || b == 0. {
        return Some(());
    }
    // Above this exponent the exact binary64 product's residual remains on
    // the subnormal lattice; below it, decline instead of hiding underflow.
    if !product.is_finite() || product.abs() < 4. * f64::MIN_POSITIVE / f64::EPSILON {
        return None;
    }
    sum.add(a.mul_add(b, -product))?;
    sum.add(product)
}

pub(super) fn witness<const D: usize>(
    state: &State<D>,
    values: &[(i32, f64)],
    rows: &mut [crate::master::root::ExactSum],
    objective: &mut crate::master::root::ExactSum,
) -> bool {
    if values.len() != state.master_variables.len() {
        return false;
    }
    for (variable, &(id, value)) in state.master_variables.iter().zip(values) {
        if id != variable.id
            || !value.is_finite()
            || value < variable.bounds.lower
            || value > variable.bounds.upper
            || variable.integer && (value.fract() != 0. || value.abs() > 9_007_199_254_740_992.)
        {
            return false;
        }
        if add_product(objective, variable.cost, value).is_none() {
            return false;
        }
        for &(row, a) in &variable.entries {
            if add_product(&mut rows[row], a, value).is_none() {
                return false;
            }
        }
    }
    true
}

impl<const D: usize> NodeSnapshot<D> {
    pub(super) fn attach_master_coefficients(
        &mut self,
        shared: usize,
        coefficients: &[(i32, f64)],
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let mut definitions = vec![];
        for &(id, a) in coefficients {
            let index = self
                .state
                .master_variables
                .iter()
                .position(|v| v.id == id)
                .ok_or_else(|| invalid("unknown shared-row master variable"))?;
            if !a.is_finite() || !seen.insert(id) {
                return Err(invalid("coalesced finite master-row coefficients required"));
            }
            definitions.push((index, a));
        }
        let physical = self.state.shared_offset + shared;
        for (index, a) in definitions {
            if a == 0. {
                continue;
            }
            self.state.master_variables[index].entries.push((shared, a));
            for (column, metadata) in self.warm.columns.iter_mut().enumerate() {
                if let Some(metadata) = metadata {
                    if let Some((v, sign)) = decode(metadata.tag) {
                        if v == index {
                            self.master.set_coefficient(physical, column, sign * a)?;
                            metadata.entries.push((physical, sign * a));
                        }
                    }
                }
            }
            for metadata in &mut self.warm.generated_pool {
                if let Some((v, sign)) = decode(metadata.tag) {
                    if v == index {
                        metadata.entries.push((physical, sign * a));
                    }
                }
            }
        }
        Ok(())
    }

    /// Add a shared additive row with independent master-variable coefficients.
    /// The caller supplies a valid restriction; original master IDs are separate
    /// from network IDs, physical arc IDs and generated-route tags.
    pub fn with_master_row(
        &self,
        row: Row,
        networks: &[RowCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(row, networks)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }

    /// Add a nonlinear floor row coupled to independent master variables.
    /// Fixed route contributions use the nonlinear coefficient, while static
    /// master columns retain their signed linear coefficients and bounds.
    pub fn with_rank_one_master_row(
        &self,
        row: Row,
        networks: &[RankOneCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_rank_one_row(row, networks)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }
}

impl<const D: usize> NodeSnapshot<D> {
    /// Add a sum of registered kernels coupled to independent master values.
    pub fn with_composite_cut_master_row(
        &self,
        row: Row,
        networks: &[RegisteredCutCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_composite_cut_row(row, networks)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }

    pub fn with_registered_cut_master_row(
        &self,
        row: Row,
        networks: &[RegisteredCutCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_registered_cut_row(row, networks)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/mp/master_variables/tests.rs"]
mod tests;
