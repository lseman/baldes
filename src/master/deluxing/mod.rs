//! Deep Lagrangian underestimate fixing for complete integer column pools.
//!
//! This is a Rust implementation of the DeLuxing procedure described by
//! Yang (2024), with RouteOpt's package used as an algorithm reference. No
//! native RouteOpt code is linked. All columns must be nonnegative integers;
//! the pool must contain every solution of interest below `cutoff`. Never use
//! this to certify fixing in an unfinished column-generation master.
//!
//! Filtering preserves solutions whose objective is at most `cutoff`. Indices
//! refer to the source matrix, which is never modified. Auxiliary LPs use an
//! independent backend supplied by the caller.
use super::{LpMaster, LpSolveOutcome, RowSense};
use crate::{invalid, Result};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct DeluxingOptions {
    pub clusters: usize,
    /// Use Lloyd clustering up to this pool size, norm quantiles above it.
    pub clustering_limit: usize,
    /// Continue a cluster's deep rounds while more columns than this are fixed.
    pub min_extra_fixes: usize,
    pub time_limit: Duration,
    /// Absolute safety margin: fixing requires a bound strictly above cutoff.
    pub tolerance: f64,
}
impl Default for DeluxingOptions {
    fn default() -> Self {
        Self {
            clusters: 10,
            clustering_limit: 10_000,
            min_extra_fixes: 10,
            time_limit: Duration::from_secs(30),
            tolerance: 1e-6,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeluxingOutput {
    /// Sorted original column indices, safe to fix to zero below the cutoff.
    pub fixed_columns: Vec<usize>,
    pub initial_fixes: usize,
    pub auxiliary_solves: usize,
    /// An LP failed to reach optimality or the wall-time budget was exhausted.
    /// Fixes certified by earlier rounds remain valid.
    pub stopped_early: bool,
    pub seconds: f64,
}

struct Column {
    cost: f64,
    upper: f64,
    entries: Vec<(usize, f64)>,
}
fn reduced_costs(columns: &[Column], dual: &[f64]) -> Vec<f64> {
    columns
        .iter()
        .map(|c| c.cost - c.entries.iter().map(|&(r, a)| dual[r] * a).sum::<f64>())
        .collect()
}
// Project approximate LP multipliers onto the row-sign cone. Column dual
// feasibility is handled explicitly by the finite-bound correction below.
fn multipliers<M: LpMaster>(model: &M, senses: &[RowSense]) -> Option<Vec<f64>> {
    if model.duals().len() != senses.len() || model.duals().iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(
        model
            .duals()
            .iter()
            .zip(senses)
            .map(|(&p, s)| match s {
                RowSense::LessEqual => p.min(0.),
                RowSense::GreaterEqual => p.max(0.),
                RowSense::Equal => p,
            })
            .collect(),
    )
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Filter a complete minimization model with zero lower bounds and integer
/// columns. `cutoff` includes the source's objective offset. `new_master` must
/// return an empty independent LP model. Finite upper bounds are required;
/// integer upper bounds are floored exactly as in the MIP export.
///
/// A time limit or unusable auxiliary LP returns the fixes already certified.
/// A solver error is propagated. Neither case mutates the source model.
pub fn filter<M: LpMaster, A: LpMaster>(
    source: &M,
    cutoff: f64,
    options: &DeluxingOptions,
    mut new_master: impl FnMut() -> Result<A>,
) -> Result<DeluxingOutput> {
    let started = Instant::now();
    if !cutoff.is_finite()
        || !source.objective_offset().is_finite()
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.
        || options.clusters == 0
    {
        return Err(invalid(
            "DeLuxing requires finite cutoff/offset, positive tolerance and clusters",
        ));
    }
    let n = source.costs().len();
    let rhs = source.rhs();
    let senses = source.senses();
    if source.upper().len() != n || senses.len() != rhs.len() || rhs.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("DeLuxing source dimensions or RHS"));
    }
    let mut columns = Vec::with_capacity(n);
    for j in 0..n {
        let (cost, upper) = (source.costs()[j], source.upper()[j]);
        let entries = source.column_entries(j)?;
        if !cost.is_finite()
            || !upper.is_finite()
            || upper < 0.
            || entries
                .iter()
                .any(|&(r, a)| r >= rhs.len() || !a.is_finite())
        {
            return Err(invalid("DeLuxing source column"));
        }
        columns.push(Column {
            cost,
            upper: upper.floor(),
            entries,
        });
    }
    let mut out = DeluxingOutput::default();
    let mut fixed: Vec<_> = columns.iter().map(|c| c.upper < 1.).collect();
    let mut solve = |rhs: &[f64], pool: &[usize], dummy: bool| -> Result<Option<Vec<f64>>> {
        let remaining = options.time_limit.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Ok(None);
        }
        let mut lp = new_master()?;
        lp.set_tolerances(1e-9, 1e-9)?;
        for (&s, &b) in senses.iter().zip(rhs) {
            lp.add_row(s, b, &[])?;
        }
        for &j in pool {
            let c = &columns[j];
            let k = lp.add_column(c.cost, &c.entries)?;
            lp.set_upper(k, c.upper)?;
        }
        if dummy {
            // Original DeLuxing's negative unit columns cap row multipliers.
            // Their upper bounds affect only the choice of multipliers; all
            // fixing certificates are recomputed on the original matrix.
            for r in 0..rhs.len() {
                lp.add_column(cutoff.abs().max(1.), &[(r, -1.)])?;
            }
        }
        if lp.solve_with_time_limit(options.time_limit.saturating_sub(started.elapsed()))?
            != LpSolveOutcome::Optimal
        {
            return Ok(None);
        }
        Ok(multipliers(&lp, senses))
    };
    let pool: Vec<_> = (0..n).filter(|&j| !fixed[j]).collect();
    if pool.is_empty() {
        out.fixed_columns = (0..n).collect();
        out.initial_fixes = n;
        out.seconds = started.elapsed().as_secs_f64();
        return Ok(out);
    }
    let Some(dual) = solve(rhs, &pool, false)? else {
        out.stopped_early = true;
        out.fixed_columns = (0..n).filter(|&j| fixed[j]).collect();
        out.seconds = started.elapsed().as_secs_f64();
        return Ok(out);
    };
    let rc = reduced_costs(&columns, &dual);
    let base = source.objective_offset()
        + dot(rhs, &dual)
        + pool
            .iter()
            .map(|&j| rc[j].min(0.) * columns[j].upper)
            .sum::<f64>();
    let threshold = cutoff + options.tolerance;
    if !base.is_finite() || rc.iter().any(|x| !x.is_finite()) {
        out.stopped_early = true;
    } else {
        for &j in &pool {
            if base + rc[j].max(0.) > threshold {
                fixed[j] = true;
            }
        }
    }
    out.initial_fixes = fixed.iter().filter(|&&x| x).count();
    // Every improving integer solution uses at most one high column, counted
    // with multiplicity: two reduced costs above half the initial gap exceed UB.
    let high: Vec<_> = rc
        .iter()
        .map(|&r| r > ((threshold - base) / 2.).max(0.))
        .collect();
    let labels = clustering(&columns, rhs, options, started);
    if !out.stopped_early {
        'clusters: for cluster in 0..options.clusters.min(n) {
            let mut target: Vec<_> = (0..n)
                .filter(|&j| !fixed[j] && labels[j] == cluster)
                .collect();
            if target.is_empty() {
                continue;
            }
            let mut first_round = true;
            loop {
                let mut residual = vec![0.; rhs.len()];
                for &j in &target {
                    for (v, &b) in residual.iter_mut().zip(rhs) {
                        *v += b;
                    }
                    for &(r, a) in &columns[j].entries {
                        residual[r] -= a;
                    }
                }
                if residual.iter().any(|x| !x.is_finite()) {
                    out.stopped_early = true;
                    break 'clusters;
                }
                let low: Vec<_> = (0..n).filter(|&j| !fixed[j] && !high[j]).collect();
                out.auxiliary_solves += 1;
                let Some(dual) = solve(&residual, &low, true)? else {
                    out.stopped_early = true;
                    break 'clusters;
                };
                let rc = reduced_costs(&columns, &dual);
                let base = source.objective_offset()
                    + dot(rhs, &dual)
                    + low
                        .iter()
                        .map(|&j| rc[j].min(0.) * columns[j].upper)
                        .sum::<f64>();
                let min_high = (0..n)
                    .filter(|&j| !fixed[j] && high[j])
                    .map(|j| rc[j])
                    .fold(0., f64::min);
                if !base.is_finite() || rc.iter().any(|x| !x.is_finite()) {
                    out.stopped_early = true;
                    break 'clusters;
                }
                let mut newly_fixed = vec![];
                for j in 0..n {
                    if fixed[j] {
                        continue;
                    }
                    let bound = if high[j] {
                        base + rc[j]
                    } else {
                        base + min_high + rc[j].max(0.)
                    };
                    if bound > threshold {
                        newly_fixed.push(j);
                    }
                }
                for &j in &newly_fixed {
                    fixed[j] = true;
                }
                let extra = newly_fixed.len();
                // The first deep RHS uses fixes outside this cluster. Later
                // RHS vectors use all fixes from the preceding deep round.
                target = newly_fixed
                    .into_iter()
                    .filter(|&j| !first_round || labels[j] != cluster)
                    .collect();
                first_round = false;
                if extra <= options.min_extra_fixes {
                    break;
                }
            }
        }
    }
    out.fixed_columns = (0..n).filter(|&j| fixed[j]).collect();
    out.seconds = started.elapsed().as_secs_f64();
    Ok(out)
}

fn clustering(
    columns: &[Column],
    rhs: &[f64],
    options: &DeluxingOptions,
    started: Instant,
) -> Vec<usize> {
    let n = columns.len();
    let k = options.clusters.min(n);
    let point = |j: usize| {
        let mut p = rhs.to_vec();
        for &(r, a) in &columns[j].entries {
            p[r] -= a;
        }
        p
    };
    let mut order: Vec<_> = (0..n)
        .map(|j| {
            let p = point(j);
            (dot(&p, &p), j)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut labels = vec![0; n];
    for (rank, &(_, j)) in order.iter().enumerate() {
        labels[j] = rank * k / n;
    }
    if n > options.clustering_limit || k == 0 {
        return labels;
    }
    // Deterministic quantile seeds, bounded Lloyd iterations. Sparse points
    // are materialized one at a time, avoiding an Ncolumns x Nrows matrix.
    let mut means: Vec<_> = (0..k).map(|c| point(order[c * n / k].1)).collect();
    for _ in 0..100 {
        if started.elapsed() >= options.time_limit {
            break;
        }
        let mut sums = vec![vec![0.; rhs.len()]; k];
        let mut counts = vec![0; k];
        let mut changed = false;
        for (j, label) in labels.iter_mut().enumerate() {
            if started.elapsed() >= options.time_limit {
                return labels;
            }
            let p = point(j);
            let nearest = means
                .iter()
                .enumerate()
                .map(|(c, m)| {
                    (
                        c,
                        p.iter().zip(m).map(|(x, y)| (x - y).powi(2)).sum::<f64>(),
                    )
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .unwrap()
                .0;
            changed |= *label != nearest;
            *label = nearest;
            counts[nearest] += 1;
            for (s, x) in sums[nearest].iter_mut().zip(p) {
                *s += x;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for (m, s) in means[c].iter_mut().zip(&sums[c]) {
                    *m = s / counts[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    labels
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/deluxing/tests.rs"]
mod tests;
