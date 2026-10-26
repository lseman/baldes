//! Necessary zero-penalty implications over the original network domain.
use super::*;
use crate::core::rounding::{add_down, multiply_down};
use crate::master::preprocessing::Bounds;

/// Return aggregate arc activities proved zero. No route enumeration is used.
/// A positive arc consumption step supplies resource-load implications.
/// Presence and registered rows also use bounded zero-coefficient walk support.
/// Both proofs retain validity when routes repeat vertices or physical arcs.
pub(in crate::master::model) fn zero_arcs<const D: usize>(
    state: &State<D>,
    limits: &[Bounds],
    starts: &[usize],
    budget: usize,
) -> (Vec<usize>, bool, usize) {
    let mut work = 0usize;
    let mut zeros = HashSet::new();
    macro_rules! tick {
        () => {{
            if work >= budget || Instant::now() >= state.deadline {
                return (zeros.into_iter().collect(), false, work);
            }
            work += 1;
        }};
    }
    let mut rows = std::collections::BTreeSet::new();
    for oracle in &state.oracles {
        for &(_, row) in oracle
            .knapsack_rows
            .iter()
            .chain(&oracle.presence_rows)
            .chain(&oracle.nonlinear_rows)
        {
            tick!();
            rows.insert(row);
        }
    }
    for row in rows {
        tick!();
        if state.shared_rows[row].sense == RowSense::GreaterEqual {
            continue;
        }
        let mut nonnegative = true;
        for oracle in &state.oracles {
            for &(r, a) in oracle
                .network
                .arc_entries
                .iter()
                .flatten()
                .chain(oracle.network.route_entries.iter())
            {
                tick!();
                if r == row && a < 0. {
                    nonnegative = false;
                }
            }
        }
        if !nonnegative {
            continue;
        }
        let mut lower = Some(0.);
        for (v, variable) in state.master_variables.iter().enumerate() {
            for &(r, a) in &variable.entries {
                tick!();
                if r != row || a == 0. {
                    continue;
                }
                let endpoint = if a > 0. {
                    limits[v].lower
                } else {
                    limits[v].upper
                };
                lower = lower.and_then(|sum| add_down(sum, multiply_down(a, endpoint)?));
            }
        }
        let Some(lower) = lower else {
            continue;
        };
        // All network terms are nonnegative. Hence no positive nonlinear
        // route coefficient is possible if static terms exhaust the upper RHS.
        if lower < state.rhs[state.shared_offset + row] {
            continue;
        }
        for (oracle, &start) in state.oracles.iter().zip(starts) {
            let mut registered = vec![];
            for &(cut, r) in &oracle.nonlinear_rows {
                tick!();
                if r == row {
                    registered.push(&oracle.solver.cuts.cuts[cut]);
                }
            }
            if !registered.is_empty() {
                let (supported_zeros, spent) = super::super::nonlinear::propagation::zero_arcs(
                    &oracle.network.graph,
                    &registered,
                    |a| limits[start + a].upper == 0. || zeros.contains(&(start + a)),
                    budget - work,
                    state.deadline,
                );
                work += spent;
                let Some(supported_zeros) = supported_zeros else {
                    return (zeros.into_iter().collect(), false, work);
                };
                zeros.extend(supported_zeros.into_iter().map(|a| start + a));
            }
            for &(input, r) in &oracle.presence_rows {
                tick!();
                if r != row {
                    continue;
                }
                let input = &oracle.solver.presence_rows()[input];
                for (a, events) in input.arc_events.iter().enumerate() {
                    tick!();
                    // Presence is monotone. A pair emitted by this one arc
                    // remains satisfied in every route containing the arc.
                    for &(left, right) in input.row.pairs() {
                        tick!();
                        let mut has_left = false;
                        let mut has_right = false;
                        for &event in events {
                            tick!();
                            has_left |= event == left;
                            has_right |= event == right;
                        }
                        if has_left && has_right {
                            zeros.insert(start + a);
                            break;
                        }
                    }
                }
                let (supported_zeros, spent) = super::super::presence_rows::propagation::zero_arcs(
                    &oracle.network.graph,
                    input,
                    |a| limits[start + a].upper == 0. || zeros.contains(&(start + a)),
                    budget - work,
                    state.deadline,
                );
                work += spent;
                let Some(supported_zeros) = supported_zeros else {
                    return (zeros.into_iter().collect(), false, work);
                };
                zeros.extend(supported_zeros.into_iter().map(|a| start + a));
            }
            for &(cut, r) in &oracle.knapsack_rows {
                if r != row {
                    continue;
                }
                let cut = &oracle.solver.cuts.knapsack_cuts[cut];
                let &(resource, scale) = &cut.graph_resources[&oracle.network.id];
                // This zero-arc inference uses the initial amount as a lower
                // envelope for every future path prefix. A negative
                // increment on the axis invalidates that shortcut, while
                // ordinary pricing and exact route replay still support it.
                if oracle
                    .network
                    .graph
                    .arcs
                    .iter()
                    .any(|arc| arc.consumption[resource] < 0.)
                {
                    continue;
                }
                for (a, arc) in oracle.network.graph.arcs.iter().enumerate() {
                    tick!();
                    let g = &oracle.network.graph;
                    let initial =
                        g.initial[resource].unwrap_or(g.vertices[g.source].bounds[resource].lower);
                    let guaranteed = (initial + arc.consumption[resource])
                        .max(g.vertices[arc.head].bounds[resource].lower)
                        .max(arc.bounds[resource].map_or(0., |w| w.lower));
                    let load = (guaranteed * scale + 1e-6).floor() as i32;
                    if cut.cut.coefficient(load) > 0 {
                        zeros.insert(start + a);
                    }
                }
            }
        }
    }
    (zeros.into_iter().collect(), true, work)
}
