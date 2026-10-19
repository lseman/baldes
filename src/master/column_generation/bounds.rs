use super::*;

pub fn gap_smaller_than_tolerance(dual: f64, primal: f64, absolute: f64, relative: f64) -> bool {
    let largest = dual.abs().max(primal.abs());
    if !(dual < primal - (largest * 1e-10 + 1e-6)) {
        return true;
    }
    if (primal - dual).abs() <= absolute {
        return true;
    }
    if primal.abs() <= 1e-6 && dual.abs() <= 1e-6 {
        return relative >= 0.;
    }
    if dual * primal < -1e-6 {
        return relative >= 1.;
    }
    (primal - dual) / primal.abs() <= relative
}

pub(super) fn lagrangian<M: LpMaster>(
    master: &M,
    duals: &[f64],
    minimum: f64,
    multiplicity: f64,
) -> f64 {
    let mut value = master.objective_offset();
    for (pi, rhs) in duals.iter().zip(master.rhs().iter()) {
        value += pi * rhs;
    }
    if minimum < 0. {
        value += multiplicity * minimum;
    }
    value
}

pub(super) fn master_reduced_cost(column: &Column, duals: &[f64]) -> f64 {
    let mut value = column.cost;
    for &(r, a) in &column.entries {
        if let Some(pi) = duals.get(r) {
            value -= pi * a;
        }
    }
    value
}

pub(super) fn publish_certificate(out: &mut Output, candidate: Option<f64>) -> Result<()> {
    let bound = candidate.unwrap_or(f64::NEG_INFINITY);
    if !bound.is_finite() && bound != f64::NEG_INFINITY {
        return Err(invalid("invalid certified pricing lower bound"));
    }
    out.certified_bound = out.certified_bound.max(bound);
    out.bound = out.certified_bound;
    Ok(())
}

pub(super) fn optimality_gap(primal: f64, dual: f64) -> f64 {
    if primal.abs() <= 1e-6 && dual.abs() <= 1e-6 {
        0.
    } else if primal * dual < -1e-6 {
        1.
    } else if primal - (primal.abs().max(dual.abs()) * 1e-10 + 1e-6) > dual {
        (primal - dual) / primal.abs()
    } else {
        (dual - primal) / dual.abs()
    }
}
