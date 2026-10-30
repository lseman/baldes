//! Necessary per-route reduced-cost ceilings for improving integer solutions.
use super::*;
use crate::core::rounding::add_up;

#[derive(Debug)]
pub(in crate::master::model) enum NetworkCutoff {
    /// No residual integer route can be selected (cap below one or no walk).
    Empty { network: i32 },
    Cost {
        network: i32,
        arc_cost_lower_bounds: Vec<f64>,
        route_constant_lower_bound: f64,
        /// Inclusive bound for the additive arc part supplied to the collector.
        arc_ceiling: f64,
        /// An objective lower bound for every selection containing one copy of
        /// this route, after adding that route's exact reduced cost.
        other_contributions: f64,
    },
}

/// Derive all networks at one immutable dual point. Unknown or exhausted proof
/// work returns no plan, never a mix of proved and unproved group coverage.
/// This is a necessary condition for integer residual route quantities >=1;
/// it must not be used to prune fractional LP columns in the original domain.
pub(in crate::master::model) fn cutoff_plan<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
    incumbent: f64,
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<NetworkCutoff>> {
    if !incumbent.is_finite() || state.universe.is_some() || Instant::now() >= deadline {
        return None;
    }
    let base = master_dual_base(state, master, duals, true)?;
    let mut minima = vec![];
    let mut contributions = vec![];
    let mut joint_bounds = vec![];
    for oracle in &state.oracles {
        if Instant::now() >= deadline {
            return None;
        }
        // A residual integer quantity below one must be zero. Its source
        // control domain needs no walk scan and contributes exactly zero.
        if oracle.multiplicity_upper() < 1. {
            minima.push(None);
            contributions.push(0.);
            joint_bounds.push(None);
            continue;
        }
        let graph = &oracle.network.graph;
        // Charge the dense worst-case walk scans before invoking the existing
        // original-domain relaxation. Product-state work shares the remainder.
        let scans = graph
            .vertices
            .len()
            .checked_mul(graph.arcs.len())?
            .checked_mul(3)?;
        *work = work.checked_sub(scans.checked_add(graph.vertices.len())?)?;
        let (minimum, joint) = if !oracle.activity_rows.is_empty() {
            let weights = additive_arc_cost_lower_bounds(oracle, state.shared_offset, duals)?;
            match columns::cutoff_minimum(
                oracle,
                state.shared_offset,
                duals,
                &weights,
                work,
                deadline,
            )? {
                Some(bounds) => (Some(bounds.reduced), Some((weights, bounds.remainder))),
                None => (None, None),
            }
        } else {
            (
                network_minimum(oracle, state.shared_offset, duals, work, deadline)?,
                None,
            )
        };
        joint_bounds.push(joint);
        if minimum.is_none() && oracle.network.multiplicity.lower > 0. {
            return None;
        }
        let contribution = match minimum {
            None => 0.,
            Some(m) => multiply(
                m,
                if m < 0. {
                    oracle.multiplicity_upper()
                } else {
                    oracle.network.multiplicity.lower
                },
            )?,
        };
        minima.push(minimum);
        contributions.push(contribution);
    }
    let mut prefix = vec![base];
    for &c in &contributions {
        prefix.push(add(*prefix.last()?, c)?);
    }
    let mut suffix = vec![0.; contributions.len() + 1];
    for i in (0..contributions.len()).rev() {
        suffix[i] = add(contributions[i], suffix[i + 1])?;
    }
    let mut plan = vec![];
    for (i, oracle) in state.oracles.iter().enumerate() {
        if Instant::now() >= deadline {
            return None;
        }
        let network = oracle.network.id;
        let Some(minimum) = minima[i] else {
            plan.push(NetworkCutoff::Empty { network });
            continue;
        };
        if oracle.multiplicity_upper() < 1. {
            plan.push(NetworkCutoff::Empty { network });
            continue;
        }
        // Force one residual copy. For a negative minimum the other copies
        // use an upper count rounded upward; for a nonnegative minimum their
        // mandatory count is rounded downward. This avoids double-counting
        // the network's original lower-multiplicity contribution.
        let rest = if minimum < 0. {
            add_up(oracle.multiplicity_upper(), -1.)?
        } else {
            add(oracle.network.multiplicity.lower, -1.)?.max(0.)
        };
        let other = add(add(prefix[i], suffix[i + 1])?, multiply(minimum, rest)?)?;
        let (weights, constant) = if let Some(joint) = joint_bounds[i].take() {
            joint
        } else {
            // The arc collector sees only scalar/dual weights. Include a proved
            // lower bound on the complete coupled objective in its route constant,
            // so signed resource charges cannot hide an improving original route.
            let domain = super::super::resource_groups::residual_domain_graph(oracle).ok()?;
            let mut constant = coupled::minimum(domain.as_ref(), work, deadline)??;
            for &(input, row) in &oracle.presence_rows {
                let dual = duals[state.shared_offset + row];
                if dual > 0. {
                    // Integer coefficient bounds must not round downward when
                    // converted to binary64. Unsupported magnitudes decline.
                    if oracle.solver.presence_rows()[input]
                        .row
                        .maximum_coefficient() as u128
                        > (1_u128 << 53)
                    {
                        return None;
                    }
                    constant = subtract_product(
                        constant,
                        dual,
                        oracle.solver.presence_rows()[input]
                            .row
                            .maximum_coefficient() as f64,
                    )?;
                }
            }
            for &(row, a) in &oracle.route_entries {
                constant = subtract_product(constant, duals[row], a)?;
            }
            for &(row, a) in oracle.network.route_entries.iter() {
                constant = subtract_product(constant, duals[state.shared_offset + row], a)?;
            }
            let weights = arc_cost_lower_bounds(oracle, state.shared_offset, duals)?;
            (weights, constant)
        };
        let ceiling = add_up(add_up(incumbent, -other)?, -constant)?;
        plan.push(NetworkCutoff::Cost {
            network,
            arc_cost_lower_bounds: weights,
            route_constant_lower_bound: constant,
            arc_ceiling: ceiling,
            other_contributions: other,
        });
    }
    (Instant::now() < deadline).then_some(plan)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/bounds/cutoff/tests.rs"]
mod tests;
