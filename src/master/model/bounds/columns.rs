//! Joint original-column bounds for coupled objectives and deterministic rows.
use super::*;
use crate::core::rounding::add_up;
use crate::enumeration::{CompleteResourceRelaxation, PathDomainLimits};

use crate::master::controls::affine::Interval;
fn push(rows: &mut BTreeMap<usize, Interval>, row: usize, value: Interval) -> Option<()> {
    let previous = rows.entry(row).or_insert(Interval::point(0.));
    *previous = previous.add(value)?;
    Some(())
}

#[cfg(test)]
pub(in crate::master::model) fn physical_paths<const D: usize>(
    oracle: &Oracle<D>,
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<Vec<usize>>> {
    let domain = super::super::resource_groups::residual_domain_graph(oracle).ok()?;
    let mut relaxed = domain.as_ref().clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    // Exact zero-activity proofs restrict future positive-quantity columns,
    // even when the corresponding inactive LP row has zero dual value.
    // Heuristic and reduced-cost exclusions are deliberately not used here.
    relaxed.arcs.retain(|arc| {
        !oracle
            .proved_zero_arcs
            .get(oracle.index[&arc.id])
            .copied()
            .unwrap_or(false)
    });
    relaxed.set_coupled_resources(&[]).ok()?;
    // Reserve half the caller's remaining logical work for collection and
    // leave the rest for source replay. Coverage must follow that budget,
    // rather than impose an unrelated fixed path/scan ceiling.
    let allowance = *work / 2;
    let paths = match CompleteResourceRelaxation::collect(
        &relaxed,
        PathDomainLimits {
            max_paths: allowance,
            max_work: allowance,
            max_stored_arcs: allowance,
            deadline: Some(deadline),
        },
    ) {
        Ok(paths) => paths,
        Err(_) => {
            *work = work.checked_sub(allowance)?;
            return None;
        }
    };
    *work = work.checked_sub(paths.work_used())?;
    Some(
        paths
            .paths()
            .iter()
            .map(|path| {
                path.arc_positions()
                    .iter()
                    .map(|&p| oracle.index[&relaxed.arcs[p].id])
                    .collect()
            })
            .collect(),
    )
}

/// Stream complete conservative paths under the same shared caller budget.
/// The visitor sees each path only for the duration of the callback.
pub(in crate::master::model) fn visit_physical_paths<const D: usize>(
    oracle: &Oracle<D>,
    work: &mut usize,
    deadline: Instant,
    mut visitor: impl FnMut(
        &[usize],
        &mut usize,
    ) -> std::result::Result<(), crate::enumeration::PathDomainFailure>,
) -> Option<usize> {
    let domain = super::super::resource_groups::residual_domain_graph(oracle).ok()?;
    let mut relaxed = domain.as_ref().clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    relaxed.arcs.retain(|arc| {
        !oracle
            .proved_zero_arcs
            .get(oracle.index[&arc.id])
            .copied()
            .unwrap_or(false)
    });
    relaxed.set_coupled_resources(&[]).ok()?;
    let allowance = *work / 2;
    let mut visitor_work = work.checked_sub(allowance)?;
    let stats = match CompleteResourceRelaxation::visit(
        &relaxed,
        PathDomainLimits {
            max_paths: allowance,
            max_work: allowance,
            max_stored_arcs: allowance,
            deadline: Some(deadline),
        },
        |path| {
            let Some(mapped) = path
                .iter()
                .map(|&position| {
                    relaxed
                        .arcs
                        .get(position)
                        .and_then(|arc| oracle.index.get(&arc.id))
                        .copied()
                })
                .collect::<Option<Vec<_>>>()
            else {
                return Err(crate::enumeration::PathDomainFailure::InvalidGraph);
            };
            visitor(&mapped, &mut visitor_work)
        },
    ) {
        Ok(stats) => stats,
        Err(_) => {
            *work = visitor_work;
            return None;
        }
    };
    *work = visitor_work.checked_add(allowance.checked_sub(stats.work_used)?)?;
    Some(stats.paths)
}

pub(super) fn minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    if super::super::control_pricing::needed(oracle) {
        return super::super::control_pricing::minimum(oracle, offset, duals, work, deadline);
    }
    let mut minimum: Option<f64> = None;
    visit_physical_paths(oracle, work, deadline, |positions, work| {
        let Some(bound) = column(oracle, offset, duals, positions, work, deadline) else {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        };
        if let Some(lower) = bound {
            minimum = Some(minimum.map_or(lower, |old| old.min(lower)));
        }
        Ok(())
    })?;
    (Instant::now() < deadline).then_some(minimum)
}

pub(super) struct CutoffMinimum {
    pub reduced: f64,
    pub remainder: f64,
}
/// Joint bounds at one dual point. Any chosen additive weights are safe only
/// with a remainder proved for every path/control in the original relaxation.
pub(super) fn cutoff_minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    weights: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<CutoffMinimum>> {
    if weights.len() != oracle.network.graph.arcs.len() {
        return None;
    }
    let mut minimum: Option<CutoffMinimum> = None;
    let mut iterations = 100_000;
    visit_physical_paths(oracle, work, deadline, |path, work| {
        let bound = if super::super::control_pricing::needed(oracle) {
            super::super::control_pricing::column_bound_using(
                oracle,
                offset,
                duals,
                path,
                work,
                &mut iterations,
                deadline,
            )
        } else {
            column(oracle, offset, duals, path, work, deadline)
        };
        let Some(bound) = bound else {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        };
        let Some(reduced) = bound else {
            return Ok(());
        };
        *work = work
            .checked_sub(
                path.len()
                    .checked_add(1)
                    .ok_or(crate::enumeration::PathDomainFailure::WorkLimit)?,
            )
            .ok_or(crate::enumeration::PathDomainFailure::WorkLimit)?;
        let mut sum = 0.;
        for &arc in path {
            let Some(weight) = weights.get(arc) else {
                return Err(crate::enumeration::PathDomainFailure::InvalidGraph);
            };
            let Some(next) = crate::core::rounding::add_up(sum, *weight) else {
                return Err(crate::enumeration::PathDomainFailure::Arithmetic);
            };
            sum = next;
        }
        let Some(remainder) = crate::core::rounding::add_down(reduced, -sum) else {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        };
        if !reduced.is_finite() || !remainder.is_finite() {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        }
        match &mut minimum {
            Some(old) => {
                old.reduced = old.reduced.min(reduced);
                old.remainder = old.remainder.min(remainder);
            }
            None => minimum = Some(CutoffMinimum { reduced, remainder }),
        }
        Ok(())
    })?;
    (Instant::now() < deadline).then_some(minimum)
}

/// Enclose a whole original column, not a rounded aggregate pricing weight.
/// `None` is a proven coupled-infeasible path; outer None is Unknown.
pub(in crate::master::model) struct ColumnTerms {
    pub cost: Interval,
    pub coupled: Interval,
    pub physical: Interval,
    pub rows: BTreeMap<usize, Interval>,
}
pub(in crate::master::model) struct SourceColumn<'a> {
    pub kernel: &'a crate::graph::coupled::CoupledKernel,
    pub controls: &'a [crate::graph::coupled::PairedControl],
    pub variable_resources: &'a [usize],
}
pub(in crate::master::model) fn column<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    if super::super::control_pricing::needed(oracle) {
        return super::super::control_pricing::column_bound(
            oracle, offset, duals, arcs, work, deadline,
        );
    }
    let source = SourceColumn {
        kernel: oracle.network.graph.coupled_kernel()?,
        controls: &[],
        variable_resources: &[],
    };
    let terms = components(oracle, offset, duals, arcs, work, deadline, source)?;
    let Some(mut terms) = terms else {
        return Some(None);
    };
    for (r, value) in terms.rows {
        terms.cost = terms.cost.add(value.scale(-*duals.get(r)?)?)?;
    }
    Some(Some(terms.cost.lo))
}

pub(in crate::master::model) fn components<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    deadline: Instant,
    source: SourceColumn<'_>,
) -> Option<Option<ColumnTerms>> {
    if Instant::now() >= deadline {
        return None;
    }
    let width = arcs.len().checked_add(1)?;
    let rows_work = width
        .checked_mul(
            duals
                .len()
                .checked_add(oracle.activity_rows.len())?
                .checked_add(1)?,
        )?
        .checked_mul(8)?;
    *work = work.checked_sub(rows_work.checked_add(width.checked_mul(width)?)?)?;
    let graph = &oracle.network.graph;
    let kernel = source.kernel;
    let (feasible, coupled) =
        kernel.certify_with_controls(arcs, source.controls, work, Some(deadline))?;
    if !feasible {
        return Some(None);
    }
    // The master reconstructs physical original costs in sorted arc-ID order.
    // Directed replay in that same order encloses the stored column cost too.
    let mut sorted = arcs.to_vec();
    sorted.sort_by_key(|&p| graph.arcs[p].id);
    let mut cost = Interval::point(0.);
    for &p in &sorted {
        cost = cost.add(Interval::point(graph.arcs[p].cost))?;
    }
    let physical = cost;
    cost = cost.add(Interval {
        lo: coupled[0],
        hi: coupled[1],
    })?;
    let mut rows = BTreeMap::new();
    for &(r, a) in &oracle.route_entries {
        push(&mut rows, r, Interval::point(a))?;
    }
    for &(r, a) in oracle.network.route_entries.iter() {
        push(&mut rows, offset + r, Interval::point(a))?;
    }
    for &p in arcs {
        for &(r, a) in &oracle.network.arc_entries[p] {
            push(&mut rows, offset + r, Interval::point(a))?;
        }
    }
    // Registered kernels have nonnegative integer coefficients. Independent
    // upper envelopes cover composite carries without trusting priced labels.
    let mut registered = BTreeMap::new();
    for &(cut, row) in &oracle.nonlinear_rows {
        let mut upper = 0.;
        for &p in arcs {
            upper = add_up(upper, super::super::nonlinear::arc_upper(oracle, cut, p))?;
        }
        push(&mut registered, row, Interval { lo: 0., hi: upper })?;
    }
    for (r, value) in registered {
        push(&mut rows, offset + r, value)?;
    }
    for &(cut, row) in &oracle.knapsack_rows {
        let (lo, hi) = oracle.solver.cuts.knapsack_cuts[cut]
            .cut
            .coefficients
            .values()
            .fold((0, 0), |(lo, hi), &value| (lo.min(value), hi.max(value)));
        push(
            &mut rows,
            offset + row,
            Interval {
                lo: lo as f64,
                hi: hi as f64,
            },
        )?;
    }
    for &(input, row) in &oracle.presence_rows {
        let upper = oracle.solver.presence_rows()[input]
            .row
            .maximum_coefficient();
        if upper as u128 > 1_u128 << 53 {
            return None;
        }
        push(
            &mut rows,
            offset + row,
            Interval {
                lo: 0.,
                hi: upper as f64,
            },
        )?;
    }
    for &(row, _) in &oracle.resource_groups {
        push(&mut rows, offset + row, Interval { lo: 0., hi: 1. })?;
    }
    let mut activities = BTreeMap::new();
    for c in oracle
        .activity_rows
        .iter()
        .filter(|c| c.coefficient != 0. && !source.variable_resources.contains(&c.resource))
    {
        let (valid, activity) =
            kernel.certify_accumulated_activity(c.resource, arcs, work, Some(deadline))?;
        if !valid {
            return Some(None);
        }
        push(
            &mut activities,
            c.row,
            Interval {
                lo: activity[0],
                hi: activity[1],
            }
            .scale(c.coefficient)?,
        )?;
    }
    for (r, value) in activities {
        push(&mut rows, offset + r, value)?;
    }
    (Instant::now() < deadline).then_some(Some(ColumnTerms {
        cost,
        coupled: Interval {
            lo: coupled[0],
            hi: coupled[1],
        },
        physical,
        rows,
    }))
}

/// Exact incumbent/fixed-reduction contributions require a singleton source
/// enclosure, not merely the numerical intrinsic activity used by the LP.
pub(in crate::master::model) fn exact_activity_entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    deadline: Instant,
) -> Option<()> {
    exact_activity_entries_using(oracle, route, &mut 1_000_000, deadline)
}
pub(in crate::master::model) fn exact_activity_entries_using<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    work: &mut usize,
    deadline: Instant,
) -> Option<()> {
    let mut rows = BTreeMap::new();
    for c in &oracle.activity_rows {
        if c.coefficient == 0. {
            continue;
        }
        let activity = super::super::activity_rows::certificate_value(
            oracle, route, c.resource, work, deadline,
        )?;
        push(
            &mut rows,
            c.row,
            Interval {
                lo: activity[0],
                hi: activity[1],
            }
            .scale(c.coefficient)?,
        )?;
    }
    for (row, value) in super::super::activity_rows::entries(oracle, route).ok()? {
        let interval = rows.get(&row)?;
        if interval.lo != value || interval.hi != value {
            return None;
        }
    }
    Some(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/activity_rows_tests.rs"]
mod tests;
