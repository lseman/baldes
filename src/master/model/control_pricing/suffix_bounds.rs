// Conservative suffix bounds and terminal objective envelopes.
use super::*;
pub(super) struct PhysicalSuffixBounds {
    pub(super) suffix: SuffixTable,
    pub(super) arc_costs: Vec<f64>,
    pub(super) route_constant: f64,
    pub(super) other_coupled_lower: f64,
    pub(super) paired_terminal_charges: Vec<PairedTerminalCharge>,
    pub(super) exact_additive: bool,
}

/// Walk bounds indexed by remaining hops. Once the recurrence is bitwise
/// stable, all subsequent rows are equal and need not be stored separately.
pub(super) struct SuffixTable {
    pub(super) rows: Vec<Vec<Option<f64>>>,
    pub(super) hops: usize,
    pub(super) stable: bool,
}

impl SuffixTable {
    pub(super) fn len(&self) -> usize {
        self.hops + 1
    }

    pub(super) fn get(&self, remaining: usize) -> Option<&Vec<Option<f64>>> {
        if remaining > self.hops {
            return None;
        }
        self.rows
            .get(remaining)
            .or_else(|| if self.stable { self.rows.last() } else { None })
    }
}

#[derive(Clone)]
pub(super) struct PairedTerminalArcCharge {
    pub(super) arc: usize,
    pub(super) values: Interval,
    pub(super) floor: f64,
    pub(super) lower: f64,
}

#[derive(Clone)]
pub(super) struct PairedTerminalCharge {
    pub(super) resource: usize,
    pub(super) normalization: crate::graph::coupled::PairedCostNormalization,
    pub(super) objective_coefficient: f64,
    pub(super) positive_direction: bool,
    pub(super) lower_monotone: bool,
    pub(super) upper_unbounded: bool,
    pub(super) baseline: Option<[f64; 2]>,
    pub(super) max_increment: f64,
    pub(super) max_waiting: f64,
    pub(super) max_lower: f64,
    pub(super) max_upper: f64,
    pub(super) terminals: Vec<PairedTerminalArcCharge>,
}

#[derive(Clone, Copy)]
pub(super) struct PairedTerminalUpperEnvelope {
    pub(super) terminal: f64,
    pub(super) max_increment: f64,
    pub(super) max_waiting: f64,
    pub(super) max_lower: f64,
    pub(super) max_upper: f64,
    pub(super) upper_unbounded: bool,
}

pub(super) fn row_simple_supported<const D: usize>(oracle: &Oracle<D>) -> bool {
    let graph = &oracle.network.graph;
    let Some(kernel) = graph.coupled_kernel() else {
        return false;
    };
    oracle.activity_rows.iter().all(|entry| {
        activity_rows::paired(graph, entry.resource)
            || matches!(
                kernel.definitions().get(entry.resource),
                Some(
                    CoupledResource::CumulativeCost { .. } | CoupledResource::CustomResource { .. }
                )
            )
    }) && (0..oracle.solver.cuts.cuts.len()).all(|cut| {
        oracle
            .nonlinear_rows
            .iter()
            .any(|&(mapped, _)| mapped == cut)
    }) && (0..oracle.solver.cuts.knapsack_cuts.len()).all(|cut| {
        oracle
            .knapsack_rows
            .iter()
            .any(|&(mapped, _)| mapped == cut)
    }) && (0..oracle.solver.presence_rows().len()).all(|input| {
        oracle
            .presence_rows
            .iter()
            .any(|&(mapped, _)| mapped == input)
    })
}

/// Route-independent upper envelope for one CostBounded paired transition.
/// It encloses the largest possible increase in Gamma and the largest lower
/// and upper function values over every valid input coordinate.
pub(super) fn paired_bounded_step_envelope(
    definition: &CoupledResource,
    arc_index: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<(f64, f64, Interval, Interval)> {
    let (arrival, increment, lower, upper, waiting) = match definition {
        CoupledResource::PairedPiecewiseLinear { arcs, mode, .. }
            if *mode == crate::graph::coupled::PairedResourceMode::CostBounded =>
        {
            let arc = arcs.get(arc_index)?;
            (
                piecewise_value_hull(&arc.arrival, work, deadline)?,
                piecewise_value_hull(&arc.increment, work, deadline)?,
                piecewise_value_hull(&arc.lower, work, deadline)?,
                piecewise_value_hull(&arc.upper, work, deadline)?,
                arc.waiting,
            )
        }
        CoupledResource::SlopePaired {
            arcs,
            mode,
            policy: crate::graph::coupled::FrontOperationPolicy::StrictDomain,
            ..
        } if *mode == crate::graph::coupled::PairedResourceMode::CostBounded => {
            let arc = arcs.get(arc_index)?;
            (
                slope_function_hull(&arc.arrival, work, deadline)?,
                slope_function_hull(&arc.increment, work, deadline)?,
                slope_function_hull(&arc.lower, work, deadline)?,
                slope_function_hull(&arc.upper, work, deadline)?,
                arc.waiting,
            )
        }
        _ => return None,
    };
    let max_head = match definition {
        CoupledResource::PairedPiecewiseLinear { arcs, .. } => {
            let arc = arcs.get(arc_index)?;
            intersect_head_domains(piecewise_domain(&arc.lower)?, piecewise_domain(&arc.upper)?)?.hi
        }
        CoupledResource::SlopePaired { arcs, .. } => {
            let arc = arcs.get(arc_index)?;
            intersect_head_domains(slope_domain(&arc.lower)?, slope_domain(&arc.upper)?)?.hi
        }
        _ => return None,
    };
    let maximum_wait = crate::core::rounding::add_up(max_head, -arrival.lo)?.max(0.);
    let waiting_increase = multiply_upper_nonnegative(f64::from(waiting), maximum_wait)?;
    (increment.hi >= 0. && lower.hi <= upper.hi).then_some((
        increment.hi,
        waiting_increase,
        lower,
        upper,
    ))
}

pub(super) fn multiply_upper_nonnegative(left: f64, right: f64) -> Option<f64> {
    if !left.is_finite() || !right.is_finite() || left < 0. || right < 0. {
        return None;
    }
    Some(Interval::point(left).scale(right)?.hi)
}

pub(super) fn paired_terminal_ceiling_after_hops(
    mut ceiling: f64,
    max_increment: f64,
    max_waiting: f64,
    max_lower: f64,
    max_upper: f64,
    upper_unbounded: bool,
    remaining: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let mut maximum = ceiling;
    for _ in 0..remaining {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = ceiling;
        ceiling = crate::core::rounding::add_up(ceiling, max_increment)?;
        ceiling = crate::core::rounding::add_up(ceiling, max_waiting)?.max(max_lower);
        // The unbounded-shift sentinel is not a terminal cap. A finite upper
        // function still caps ordinary bounded resources.
        if !upper_unbounded {
            ceiling = ceiling.min(max_upper);
        }
        maximum = maximum.max(ceiling);
        // All steps are monotone in the previous ceiling. Once the state
        // decreases, later states cannot increase the accumulated maximum.
        if ceiling <= previous {
            break;
        }
    }
    Some(maximum)
}

/// Direction in which an isolated paired reduced-cost charge moves with
/// Gamma. Every nonzero objective and activity-row term must have the same
/// sign; otherwise endpoint tightening is not valid for their sum.
pub(super) fn paired_cost_direction<const D: usize>(
    oracle: &Oracle<D>,
    resource: usize,
    objective_coefficient: f64,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<bool>> {
    let mut direction = (objective_coefficient != 0.).then_some(objective_coefficient > 0.);
    for entry in oracle
        .activity_rows
        .iter()
        .filter(|entry| entry.resource == resource && entry.coefficient != 0.)
    {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let row_dual = -*duals.get(offset.checked_add(entry.row)?)?;
        if row_dual == 0. {
            continue;
        }
        let mut row_terms = 0usize;
        for candidate in &oracle.activity_rows {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            if candidate.row == entry.row && candidate.coefficient != 0. {
                row_terms += 1;
            }
        }
        if row_terms != 1
            || !paired_activity_row_is_isolated(oracle, offset, entry.row, work, deadline)?
        {
            return Some(None);
        }
        let positive = entry.coefficient.is_sign_positive() == row_dual.is_sign_positive();
        if direction.is_some_and(|existing| existing != positive) {
            return Some(None);
        }
        direction = Some(positive);
    }
    Some(direction)
}

pub(super) fn coupled_lower_bounds<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<(f64, Vec<f64>, Vec<PairedTerminalCharge>)> {
    let graph = &oracle.network.graph;
    let kernel = graph.coupled_kernel()?;
    let mut lower = 0.;
    let mut paired_arc_lower = vec![0.; graph.arcs.len()];
    let mut paired_terminal_charges = Vec::new();
    for (resource, definition) in kernel.definitions().iter().enumerate() {
        let accumulated = match definition {
            CoupledResource::CumulativeCost {
                initial,
                increments,
                weights,
                ..
            } => Some((*initial, increments.as_slice(), weights.as_slice())),
            CoupledResource::CustomResource {
                increments,
                weights,
                ..
            } => Some((0., increments.as_slice(), weights.as_slice())),
            _ => None,
        };
        if let Some((initial, increments, weights)) = accumulated {
            let coefficient = kernel.effective_objective_coefficient(resource)?;
            let has_activity = oracle
                .activity_rows
                .iter()
                .any(|entry| entry.resource == resource && entry.coefficient != 0.);
            if coefficient != 0. || has_activity {
                let amount = accumulated_cost_hull(
                    initial,
                    increments,
                    weights,
                    graph.original_max_arcs(),
                    work,
                    deadline,
                )?;
                if coefficient != 0. {
                    lower = crate::core::rounding::add_down(lower, amount.scale(coefficient)?.lo)?;
                }
                for entry in oracle
                    .activity_rows
                    .iter()
                    .filter(|entry| entry.resource == resource && entry.coefficient != 0.)
                {
                    let dual = *duals.get(offset.checked_add(entry.row)?)?;
                    lower = crate::core::rounding::add_down(
                        lower,
                        amount.scale(entry.coefficient)?.scale(-dual)?.lo,
                    )?;
                }
            }
            continue;
        }
        let charge_hull = match definition {
            CoupledResource::IntervalCost { functions, .. }
            | CoupledResource::PiecewiseLinearCost { functions, .. } => {
                piecewise_charge_hull(functions, work, deadline)
            }
            CoupledResource::EncodedPiecewiseLinearCost { functions, .. } => {
                encoded_charge_hull(functions, work, deadline)
            }
            CoupledResource::SlopeIntervalCost { functions, .. } => {
                slope_charge_hull(functions, work, deadline)
            }
            _ => None,
        };
        if let Some(charge_hull) = charge_hull {
            let coefficient = kernel.effective_objective_coefficient(resource)?;
            if coefficient != 0. {
                let amount =
                    repeated_charge_hull(charge_hull, graph.original_max_arcs(), work, deadline)?;
                lower = crate::core::rounding::add_down(lower, amount.scale(coefficient)?.lo)?;
            }
            continue;
        }
        let (normalization, cost_mode) = match definition {
            CoupledResource::PairedPiecewiseLinear {
                initial,
                mode,
                normalization,
                arcs,
                ..
            } => {
                if *mode == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && arcs.iter().any(|arc| {
                        arc.upper
                            .points
                            .first()
                            .is_some_and(|point| point[1] >= 1e12)
                    })
                {
                    // The sentinel enables unbounded shifts. The finite-hop
                    // envelope deliberately ignores its value as a cap; the
                    // source-control replay still validates every proposal
                    // against the declared upper function. Mixed signs and
                    // non-isolated rows fall back to path pricing.
                    let objective_coefficient = kernel.effective_objective_coefficient(resource)?;
                    let direction = paired_cost_direction(
                        oracle,
                        resource,
                        objective_coefficient,
                        offset,
                        duals,
                        work,
                        deadline,
                    )?;
                    if direction == Some(false) {
                        let terminal_lower = paired_terminal_lower_bound(
                            initial,
                            arcs,
                            graph.original_max_arcs(),
                            work,
                            deadline,
                        )?;
                        let upper = paired_terminal_upper_bound(
                            initial,
                            arcs,
                            graph.original_max_arcs(),
                            work,
                            deadline,
                        )?;
                        debug_assert!(upper.upper_unbounded);
                        if terminal_lower > upper.terminal {
                            return None;
                        }
                        let baseline = if objective_coefficient != 0.
                            && *normalization
                                == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
                        {
                            Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
                        } else {
                            None
                        };
                        let values = Interval {
                            lo: terminal_lower,
                            hi: upper.terminal,
                        };
                        let mut terminals = Vec::new();
                        let mut shared_terminal_lower = None;
                        for (arc_index, arc) in graph.arcs.iter().enumerate() {
                            if arc.head != graph.sink {
                                continue;
                            }
                            let contribution_work = if shared_terminal_lower.is_some() {
                                1
                            } else {
                                oracle.activity_rows.len().checked_add(1)?
                            };
                            *work = work.checked_sub(contribution_work)?;
                            if Instant::now() >= deadline {
                                return None;
                            }
                            // This unbounded envelope is global to the
                            // resource: its values and contribution do not
                            // depend on the sink arc. Compute it once, while
                            // still charging each physical-arc update.
                            let terminal_lower_charge = match shared_terminal_lower {
                                Some(value) => value,
                                None => {
                                    let mut value = 0.;
                                    add_paired_arc_contribution(
                                        oracle,
                                        offset,
                                        duals,
                                        resource,
                                        *normalization,
                                        objective_coefficient,
                                        baseline,
                                        values,
                                        &mut value,
                                    )?;
                                    shared_terminal_lower = Some(value);
                                    value
                                }
                            };
                            paired_arc_lower[arc_index] = crate::core::rounding::add_down(
                                paired_arc_lower[arc_index],
                                terminal_lower_charge,
                            )?;
                            if terminals.is_empty() {
                                // Every sink arc has the same refinement,
                                // so its minimum is unchanged by retaining
                                // one representative. Arc-specific lower
                                // bounds are still installed above.
                                terminals.push(PairedTerminalArcCharge {
                                    arc: arc_index,
                                    values,
                                    floor: terminal_lower,
                                    lower: terminal_lower_charge,
                                });
                            }
                        }
                        if !terminals.is_empty() {
                            paired_terminal_charges.push(PairedTerminalCharge {
                                resource,
                                normalization: *normalization,
                                objective_coefficient,
                                positive_direction: false,
                                lower_monotone: false,
                                upper_unbounded: upper.upper_unbounded,
                                baseline,
                                max_increment: upper.max_increment,
                                max_waiting: upper.max_waiting,
                                max_lower: upper.max_lower,
                                max_upper: upper.max_upper,
                                terminals,
                            });
                        }
                        continue;
                    }
                    if direction.is_none() {
                        let mut has_effect = objective_coefficient != 0.;
                        for entry in oracle
                            .activity_rows
                            .iter()
                            .filter(|entry| entry.resource == resource && entry.coefficient != 0.)
                        {
                            *work = work.checked_sub(1)?;
                            if Instant::now() >= deadline {
                                return None;
                            }
                            has_effect |= *duals.get(offset.checked_add(entry.row)?)? != 0.;
                        }
                        if has_effect {
                            return None;
                        }
                        continue;
                    }
                    if direction != Some(true) {
                        return None;
                    }
                    let terminal_lower = paired_terminal_lower_bound(
                        initial,
                        arcs,
                        graph.original_max_arcs(),
                        work,
                        deadline,
                    )?;
                    let baseline = if objective_coefficient != 0.
                        && *normalization
                            == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
                    {
                        Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
                    } else {
                        None
                    };
                    let values = Interval::point(terminal_lower);
                    for (arc_index, arc) in graph.arcs.iter().enumerate() {
                        if arc.head != graph.sink {
                            continue;
                        }
                        let contribution_work = oracle.activity_rows.len().checked_add(1)?;
                        *work = work.checked_sub(contribution_work)?;
                        if Instant::now() >= deadline {
                            return None;
                        }
                        let mut contribution = 0.;
                        add_paired_arc_contribution(
                            oracle,
                            offset,
                            duals,
                            resource,
                            *normalization,
                            objective_coefficient,
                            baseline,
                            values,
                            &mut contribution,
                        )?;
                        paired_arc_lower[arc_index] = crate::core::rounding::add_down(
                            paired_arc_lower[arc_index],
                            contribution,
                        )?;
                    }
                    continue;
                }
                if *mode == crate::graph::coupled::PairedResourceMode::Feasibility
                    && !oracle
                        .activity_rows
                        .iter()
                        .any(|entry| entry.resource == resource)
                {
                    continue;
                }
                (
                    *normalization,
                    *mode != crate::graph::coupled::PairedResourceMode::Feasibility,
                )
            }
            CoupledResource::SlopePaired { .. } => {
                let CoupledResource::SlopePaired {
                    initial,
                    mode,
                    normalization,
                    arcs,
                    policy,
                    ..
                } = definition
                else {
                    unreachable!()
                };
                if *mode == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && *policy == crate::graph::coupled::FrontOperationPolicy::FrontAlgebra
                    && arcs
                        .first()
                        .and_then(|arc| arc.upper.pairs().first())
                        .is_some_and(|pair| pair.0 >= 1e12)
                {
                    // The sentinel enables shift dominance. Positive charges
                    // use a finite-hop floor and negative charges use a
                    // finite-hop ceiling; both bound source replay and include
                    // endpoint-tolerant waiting. Mixed signs and non-isolated
                    // rows remain on the exhaustive path visitor.
                    let objective_coefficient = kernel.effective_objective_coefficient(resource)?;
                    let direction = paired_cost_direction(
                        oracle,
                        resource,
                        objective_coefficient,
                        offset,
                        duals,
                        work,
                        deadline,
                    )?;
                    if direction == Some(true) {
                        let terminal_lower = slope_paired_terminal_lower_bound(
                            initial,
                            arcs,
                            graph.original_max_arcs(),
                            work,
                            deadline,
                        )?;
                        let baseline = if objective_coefficient != 0.
                            && *normalization
                                == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
                        {
                            Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
                        } else {
                            None
                        };
                        let values = Interval::point(terminal_lower);
                        for (arc_index, paired_arc_lower) in paired_arc_lower.iter_mut().enumerate()
                        {
                            if graph.arcs.get(arc_index)?.head != graph.sink {
                                continue;
                            }
                            let contribution_work = oracle.activity_rows.len().checked_add(1)?;
                            *work = work.checked_sub(contribution_work)?;
                            if Instant::now() >= deadline {
                                return None;
                            }
                            let mut terminal_lower_charge = 0.;
                            add_paired_arc_contribution(
                                oracle,
                                offset,
                                duals,
                                resource,
                                *normalization,
                                objective_coefficient,
                                baseline,
                                values,
                                &mut terminal_lower_charge,
                            )?;
                            *paired_arc_lower = crate::core::rounding::add_down(
                                *paired_arc_lower,
                                terminal_lower_charge,
                            )?;
                        }
                        continue;
                    }
                    if direction == Some(false) {
                        let terminal_upper = slope_paired_terminal_upper_bound(
                            initial,
                            arcs,
                            graph.original_max_arcs(),
                            work,
                            deadline,
                        )?;
                        let baseline = if objective_coefficient != 0.
                            && *normalization
                                == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
                        {
                            Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
                        } else {
                            None
                        };
                        let values = Interval::point(terminal_upper);
                        let mut shared_terminal_upper = None;
                        for (arc_index, paired_arc_lower) in paired_arc_lower.iter_mut().enumerate()
                        {
                            if graph.arcs.get(arc_index)?.head != graph.sink {
                                continue;
                            }
                            let contribution_work = if shared_terminal_upper.is_some() {
                                1
                            } else {
                                oracle.activity_rows.len().checked_add(1)?
                            };
                            *work = work.checked_sub(contribution_work)?;
                            if Instant::now() >= deadline {
                                return None;
                            }
                            // The terminal envelope and every term in this
                            // contribution are shared across the sink arcs.
                            // Only the directed addition into each arc's own
                            // lower bound depends on its physical position.
                            let terminal_upper_charge = match shared_terminal_upper {
                                Some(value) => value,
                                None => {
                                    let mut value = 0.;
                                    add_paired_arc_contribution(
                                        oracle,
                                        offset,
                                        duals,
                                        resource,
                                        *normalization,
                                        objective_coefficient,
                                        baseline,
                                        values,
                                        &mut value,
                                    )?;
                                    shared_terminal_upper = Some(value);
                                    value
                                }
                            };
                            *paired_arc_lower = crate::core::rounding::add_down(
                                *paired_arc_lower,
                                terminal_upper_charge,
                            )?;
                        }
                        continue;
                    }
                    if direction.is_none() {
                        let mut has_effect = objective_coefficient != 0.;
                        for entry in oracle
                            .activity_rows
                            .iter()
                            .filter(|entry| entry.resource == resource && entry.coefficient != 0.)
                        {
                            *work = work.checked_sub(1)?;
                            if Instant::now() >= deadline {
                                return None;
                            }
                            has_effect |= *duals.get(offset.checked_add(entry.row)?)? != 0.;
                        }
                        if !has_effect {
                            continue;
                        }
                    }
                    // Mixed signs and non-isolated shared rows need a joint
                    // row rounding envelope that this relaxation lacks.
                    return None;
                }
                if *mode == crate::graph::coupled::PairedResourceMode::Feasibility
                    && !oracle
                        .activity_rows
                        .iter()
                        .any(|entry| entry.resource == resource)
                {
                    continue;
                }
                (
                    *normalization,
                    *mode != crate::graph::coupled::PairedResourceMode::Feasibility,
                )
            }
            _ => continue,
        };
        let objective_coefficient = if cost_mode {
            kernel.effective_objective_coefficient(resource)?
        } else {
            0.
        };
        let baseline = if cost_mode
            && normalization
                == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
        {
            Some(kernel.paired_initial_minimum(resource, work, Some(deadline))?)
        } else {
            None
        };
        let is_bounded = matches!(
            definition,
            CoupledResource::PairedPiecewiseLinear {
                mode: crate::graph::coupled::PairedResourceMode::CostBounded,
                ..
            } | CoupledResource::SlopePaired {
                mode: crate::graph::coupled::PairedResourceMode::CostBounded,
                policy: crate::graph::coupled::FrontOperationPolicy::StrictDomain,
                ..
            }
        );
        let positive_direction = if is_bounded {
            paired_cost_direction(
                oracle,
                resource,
                objective_coefficient,
                offset,
                duals,
                work,
                deadline,
            )?
        } else {
            None
        };
        let monotone_cost_bounded = positive_direction.is_some();
        let mut max_increment = 0.0f64;
        let mut max_waiting = 0.0f64;
        let mut max_lower = f64::NEG_INFINITY;
        if monotone_cost_bounded {
            let mut max_upper = f64::NEG_INFINITY;
            let mut terminal_arcs = Vec::new();
            let mut terminal_contributions = HashMap::new();
            for arc_index in 0..graph.arcs.len() {
                let (increment, waiting, lower, upper) =
                    paired_bounded_step_envelope(definition, arc_index, work, deadline)?;
                max_increment = max_increment.max(increment);
                max_waiting = max_waiting.max(waiting);
                max_lower = max_lower.max(lower.hi);
                max_upper = max_upper.max(upper.hi);
                if graph.arcs.get(arc_index)?.head == graph.sink {
                    // The same hulls supply the terminal contribution. The
                    // physical suffix bound only reads paired arc values on
                    // sink arcs, so nonterminal contributions are unnecessary.
                    add_bounded_terminal_contribution(
                        oracle,
                        offset,
                        duals,
                        resource,
                        normalization,
                        objective_coefficient,
                        baseline,
                        Interval {
                            lo: lower.lo,
                            hi: upper.hi,
                        },
                        arc_index,
                        paired_arc_lower.get_mut(arc_index)?,
                        &mut terminal_arcs,
                        &mut terminal_contributions,
                        work,
                        deadline,
                    )?;
                }
            }
            if !max_increment.is_finite()
                || !max_waiting.is_finite()
                || !max_lower.is_finite()
                || !max_upper.is_finite()
            {
                return None;
            }
            if !terminal_arcs.is_empty() {
                paired_terminal_charges.push(PairedTerminalCharge {
                    resource,
                    normalization,
                    objective_coefficient,
                    positive_direction: positive_direction?,
                    lower_monotone: true,
                    upper_unbounded: false,
                    baseline,
                    max_increment,
                    max_waiting,
                    max_lower,
                    max_upper,
                    terminals: terminal_arcs,
                });
            }
            continue;
        }
        for (arc_index, paired_arc_lower) in paired_arc_lower.iter_mut().enumerate() {
            let (lower_function, upper_function) = match definition {
                CoupledResource::PairedPiecewiseLinear { arcs, .. } => {
                    let arc = arcs.get(arc_index)?;
                    (&arc.lower, &arc.upper)
                }
                CoupledResource::SlopePaired { arcs, .. } => {
                    let arc = arcs.get(arc_index)?;
                    let low = slope_function_hull(&arc.lower, work, deadline)?;
                    let high = slope_function_hull(&arc.upper, work, deadline)?;
                    let values = Interval {
                        lo: low.lo,
                        hi: high.hi,
                    };
                    add_paired_arc_contribution(
                        oracle,
                        offset,
                        duals,
                        resource,
                        normalization,
                        objective_coefficient,
                        baseline,
                        values,
                        paired_arc_lower,
                    )?;
                    continue;
                }
                _ => continue,
            };
            let lower_hull = piecewise_value_hull(lower_function, work, deadline)?;
            let upper_hull = piecewise_value_hull(upper_function, work, deadline)?;
            let values = Interval {
                lo: lower_hull.lo,
                hi: upper_hull.hi,
            };
            if values.lo > values.hi {
                return None;
            }
            add_paired_arc_contribution(
                oracle,
                offset,
                duals,
                resource,
                normalization,
                objective_coefficient,
                baseline,
                values,
                paired_arc_lower,
            )?;
        }
    }
    let hops = graph.original_max_arcs();
    for &(cut, row) in &oracle.nonlinear_rows {
        let mut per_arc = 0u64;
        for arc in 0..graph.arcs.len() {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            per_arc = per_arc.max(super::nonlinear::arc_upper(oracle, cut, arc) as u64);
        }
        let maximum = u128::try_from(hops)
            .ok()?
            .checked_mul(u128::from(per_arc))?;
        if maximum > (1u128 << 53) {
            return None;
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: 0.,
                hi: maximum as f64,
            },
            &mut lower,
        )?;
    }
    for &(cut, row) in &oracle.knapsack_rows {
        let cut = &oracle.solver.cuts.knapsack_cuts.get(cut)?.cut;
        let mut low = 0i32;
        let mut high = 0i32;
        for &value in cut.coefficients.values() {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            low = low.min(value);
            high = high.max(value);
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: f64::from(low),
                hi: f64::from(high),
            },
            &mut lower,
        )?;
    }
    for &(input, row) in &oracle.presence_rows {
        let maximum = oracle
            .solver
            .presence_rows()
            .get(input)?
            .row
            .maximum_coefficient();
        if maximum as u128 > (1u128 << 53) {
            return None;
        }
        add_master_row_lower(
            offset,
            duals,
            row,
            Interval {
                lo: 0.,
                hi: maximum as f64,
            },
            &mut lower,
        )?;
    }
    for &(row, _) in &oracle.resource_groups {
        add_master_row_lower(offset, duals, row, Interval { lo: 0., hi: 1. }, &mut lower)?;
    }
    Some((lower, paired_arc_lower, paired_terminal_charges))
}

/// Lower-bound every terminal value reachable through at most `hops` paired
/// transitions when the cost-mode upper function is the unbounded sentinel.
/// The transition's feasible value is at least both the increment/waiting
/// recurrence and the lower function. Relaxing each over its complete encoded
/// domain gives a route-independent recurrence; Unknown arithmetic keeps the
/// caller on exhaustive path pricing.
pub(super) fn paired_terminal_lower_bound(
    initial: &crate::graph::coupled::PiecewiseLinearFunction,
    arcs: &[crate::graph::coupled::PairedResourceArc],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let initial = piecewise_value_hull(initial, work, deadline)?;
    if hops == 0 {
        return Some(initial.lo);
    }
    let mut increment_lower = f64::INFINITY;
    let mut lower_function_minimum = f64::INFINITY;
    let mut minimum_waiting_delta = 0.0f64;

    for arc in arcs {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let arrival = piecewise_value_hull(&arc.arrival, work, deadline)?;
        let increment = piecewise_value_hull(&arc.increment, work, deadline)?;
        let lower = piecewise_value_hull(&arc.lower, work, deadline)?;
        let lower_head = arc.lower.points.last()?.first().copied()?;
        let upper_head = arc.upper.points.last()?.first().copied()?;
        let maximum_head = lower_head.max(upper_head);
        let maximum_wait = crate::core::rounding::add_up(maximum_head, -arrival.lo)?;
        if maximum_wait < 0. {
            return None;
        }
        let wait_delta = if arc.waiting < 0 {
            crate::core::rounding::multiply_down(f64::from(arc.waiting), maximum_wait)?
        } else {
            0.0
        };
        increment_lower = increment_lower.min(increment.lo);
        lower_function_minimum = lower_function_minimum.min(lower.lo);
        minimum_waiting_delta = minimum_waiting_delta.min(wait_delta);
    }

    let mut current = initial.lo;
    let mut terminal_lower = current;
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = current;
        let after_increment = crate::core::rounding::add_down(current, increment_lower)?;
        let recurrence = crate::core::rounding::add_down(after_increment, minimum_waiting_delta)?;
        current = recurrence.max(lower_function_minimum);
        terminal_lower = terminal_lower.min(current);
        // The directed recurrence is monotone. Once it moves upward, every
        // subsequent state is at least as large and cannot lower this bound.
        if current >= previous {
            break;
        }
    }
    terminal_lower.is_finite().then_some(terminal_lower)
}

/// Lower-bound terminal Gamma for raw-slope paired costs with an unbounded
/// upper sentinel. Function hulls cover every prepared value, including the
/// first/last values returned by FrontAlgebra endpoint snapping. The domain
/// slack below encloses every coordinate accepted by that same near rule.
pub(super) fn slope_paired_terminal_lower_bound(
    initial: &crate::graph::coupled::PreparedSlopeFunction,
    arcs: &[crate::graph::coupled::SlopeRelationArc],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let initial = slope_function_hull(initial, work, deadline)?;
    if hops == 0 {
        return Some(initial.lo);
    }
    let mut increment_lower = f64::INFINITY;
    let mut lower_function_minimum = f64::INFINITY;
    let mut minimum_waiting_delta = 0.0f64;

    for arc in arcs {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let head = intersect_head_domains(
            slope_domain_with_endpoint_tolerance(&arc.lower)?,
            slope_domain_with_endpoint_tolerance(&arc.upper)?,
        )?;
        let arrival = slope_function_hull(&arc.arrival, work, deadline)?;
        let increment = slope_function_hull(&arc.increment, work, deadline)?;
        let lower = slope_function_hull(&arc.lower, work, deadline)?;
        let head_minus_arrival = Interval {
            lo: crate::core::rounding::add_down(head.lo, -arrival.hi)?,
            hi: crate::core::rounding::add_up(head.hi, -arrival.lo)?,
        };
        let waiting_delta = Interval::point(f64::from(arc.waiting))
            .multiply(head_minus_arrival)?
            .lo;
        increment_lower = increment_lower.min(increment.lo);
        lower_function_minimum = lower_function_minimum.min(lower.lo);
        minimum_waiting_delta = minimum_waiting_delta.min(waiting_delta);
    }

    let mut current = initial.lo;
    let mut terminal_lower = current;
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = current;
        let after_increment = crate::core::rounding::add_down(current, increment_lower)?;
        let recurrence = crate::core::rounding::add_down(after_increment, minimum_waiting_delta)?;
        current = recurrence.max(lower_function_minimum);
        terminal_lower = terminal_lower.min(current);
        if current >= previous {
            break;
        }
    }
    terminal_lower.is_finite().then_some(terminal_lower)
}

/// Upper-bound the minimal source replay recurrence for raw-slope paired
/// costs with an unbounded upper sentinel. The upper function limits
/// feasibility, but the sentinel is not a cap on the recurrence; endpoint-
/// tolerant arrival/head domains remain in the envelope.
pub(super) fn slope_paired_terminal_upper_bound(
    initial: &crate::graph::coupled::PreparedSlopeFunction,
    arcs: &[crate::graph::coupled::SlopeRelationArc],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let initial = slope_function_hull(initial, work, deadline)?;
    if hops == 0 {
        return Some(initial.hi);
    }
    let mut increment_upper = f64::NEG_INFINITY;
    let mut lower_function_maximum = f64::NEG_INFINITY;
    let mut maximum_waiting_delta = f64::NEG_INFINITY;
    for arc in arcs {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let head = intersect_head_domains(
            slope_domain_with_endpoint_tolerance(&arc.lower)?,
            slope_domain_with_endpoint_tolerance(&arc.upper)?,
        )?;
        let arrival = slope_function_hull(&arc.arrival, work, deadline)?;
        let increment = slope_function_hull(&arc.increment, work, deadline)?;
        let lower = slope_function_hull(&arc.lower, work, deadline)?;
        let head_minus_arrival = Interval {
            lo: crate::core::rounding::add_down(head.lo, -arrival.hi)?,
            hi: crate::core::rounding::add_up(head.hi, -arrival.lo)?,
        };
        let waiting_delta = Interval::point(f64::from(arc.waiting))
            .multiply(head_minus_arrival)?
            .hi;
        increment_upper = increment_upper.max(increment.hi);
        lower_function_maximum = lower_function_maximum.max(lower.hi);
        maximum_waiting_delta = maximum_waiting_delta.max(waiting_delta);
    }
    if !increment_upper.is_finite()
        || !lower_function_maximum.is_finite()
        || !maximum_waiting_delta.is_finite()
    {
        return None;
    }

    let mut current = initial.hi;
    let mut terminal_upper = current;
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = current;
        current = crate::core::rounding::add_up(current, increment_upper)?;
        current = crate::core::rounding::add_up(current, maximum_waiting_delta)?;
        current = current.max(lower_function_maximum);
        terminal_upper = terminal_upper.max(current);
        // A monotone recurrence that moves downward cannot increase the
        // accumulated terminal upper bound on any later hop.
        if current <= previous {
            break;
        }
    }
    terminal_upper.is_finite().then_some(terminal_upper)
}

/// Expand a raw-slope domain enough to cover the absolute/relative endpoint
/// snapping used by FrontAlgebra. For relative snapping, if d=|x-b| then
/// d < r|x+b| <= r(2|b|+d), hence d < 2r|b|/(1-r) <= 4r|b|.
pub(super) fn slope_domain_with_endpoint_tolerance(
    function: &crate::graph::coupled::PreparedSlopeFunction,
) -> Option<(f64, f64)> {
    const RELATIVE_PADDING: f64 = 8.881784197001252e-10;
    const ABSOLUTE_TOLERANCE: f64 = 0.0001;

    let (lower, upper) = slope_domain(function)?;
    let padding = |endpoint: f64| {
        multiply_upper_nonnegative(
            RELATIVE_PADDING,
            crate::core::rounding::add_up(endpoint.abs(), 1.0)?,
        )
        .map(|relative| relative.max(ABSOLUTE_TOLERANCE))
    };
    Some((
        crate::core::rounding::add_down(lower, -padding(lower)?)?,
        crate::core::rounding::add_up(upper, padding(upper)?)?,
    ))
}

/// Upper-bound terminal Gamma through the route's finite hop limit. When any
/// arc has the 1e12 CostUnbounded marker, ignore upper-value caps in this
/// relaxation; retain function-domain checks and rely on source replay to
/// validate candidate controls. Otherwise, the finite upper hulls can cap the
/// recurrence directly.
pub(super) fn paired_terminal_upper_bound(
    initial: &crate::graph::coupled::PiecewiseLinearFunction,
    arcs: &[crate::graph::coupled::PairedResourceArc],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<PairedTerminalUpperEnvelope> {
    let initial = piecewise_value_hull(initial, work, deadline)?;
    let upper_unbounded = arcs.iter().any(|arc| {
        arc.upper
            .points
            .first()
            .is_some_and(|point| point[1] >= 1e12)
    });
    let mut max_increment = f64::NEG_INFINITY;
    let mut max_waiting = 0.0f64;
    let mut max_lower = f64::NEG_INFINITY;
    let mut max_upper = f64::NEG_INFINITY;
    let mut valid_arcs = 0usize;
    for arc in arcs {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let Some(head) =
            intersect_head_domains(piecewise_domain(&arc.lower)?, piecewise_domain(&arc.upper)?)
        else {
            continue;
        };
        let arrival = piecewise_value_hull(&arc.arrival, work, deadline)?;
        let increment = piecewise_value_hull(&arc.increment, work, deadline)?;
        let lower = piecewise_value_hull(&arc.lower, work, deadline)?;
        let upper = piecewise_value_hull(&arc.upper, work, deadline)?;
        let maximum_wait = crate::core::rounding::add_up(head.hi, -arrival.lo)?.max(0.);
        let waiting = if arc.waiting > 0 {
            multiply_upper_nonnegative(f64::from(arc.waiting), maximum_wait)?
        } else {
            0.0
        };
        max_increment = max_increment.max(increment.hi);
        max_waiting = max_waiting.max(waiting);
        max_lower = max_lower.max(lower.hi);
        max_upper = max_upper.max(upper.hi);
        valid_arcs += 1;
    }
    if valid_arcs == 0
        || !max_increment.is_finite()
        || !max_waiting.is_finite()
        || !max_lower.is_finite()
        || !max_upper.is_finite()
    {
        return None;
    }

    let mut current = initial.hi;
    let mut terminal = current;
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = current;
        current = crate::core::rounding::add_up(current, max_increment)?;
        current = crate::core::rounding::add_up(current, max_waiting)?.max(max_lower);
        if !upper_unbounded {
            current = current.min(max_upper);
        }
        terminal = terminal.max(current);
        if current <= previous {
            break;
        }
    }
    terminal.is_finite().then_some(PairedTerminalUpperEnvelope {
        terminal,
        max_increment,
        max_waiting,
        max_lower,
        max_upper,
        upper_unbounded,
    })
}

/// A sentinel activity term is separable only when no other local column
/// coefficient is accumulated into the same master row. The row's source sum
/// is rounded before its dual multiplication, so combining it with route,
/// arc, cut, or another activity term needs a joint rounding envelope.
pub(super) fn paired_activity_row_is_isolated<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    row: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<bool> {
    let global_row = offset.checked_add(row)?;
    for &(candidate_row, coefficient) in &oracle.route_entries {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if candidate_row == global_row && coefficient != 0. {
            return Some(false);
        }
    }
    for &(candidate_row, coefficient) in oracle.network.route_entries.iter() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if candidate_row == row && coefficient != 0. {
            return Some(false);
        }
    }
    for entries in oracle.network.arc_entries.iter() {
        for &(candidate_row, coefficient) in entries {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            if candidate_row == row && coefficient != 0. {
                return Some(false);
            }
        }
    }
    for &(_, candidate_row) in oracle
        .nonlinear_rows
        .iter()
        .chain(&oracle.knapsack_rows)
        .chain(&oracle.presence_rows)
    {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if candidate_row == row {
            return Some(false);
        }
    }
    for &(candidate_row, _) in &oracle.resource_groups {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if candidate_row == row {
            return Some(false);
        }
    }
    Some(true)
}

pub(super) fn add_master_row_lower(
    offset: usize,
    duals: &[f64],
    row: usize,
    values: Interval,
    lower: &mut f64,
) -> Option<()> {
    let dual = *duals.get(offset.checked_add(row)?)?;
    *lower = crate::core::rounding::add_down(*lower, values.scale(-dual)?.lo)?;
    Some(())
}

pub(super) fn add_paired_arc_contribution<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    resource: usize,
    normalization: crate::graph::coupled::PairedCostNormalization,
    objective_coefficient: f64,
    baseline: Option<[f64; 2]>,
    values: Interval,
    lower: &mut f64,
) -> Option<()> {
    if !values.lo.is_finite() || !values.hi.is_finite() || values.lo > values.hi {
        return None;
    }
    let domains = [values];
    let terminal = Expression::variable(0, 1);
    let mut reduced = Expression::constant(Interval::point(0.), 1);
    if objective_coefficient != 0. {
        let mut amount = terminal.clone();
        if normalization == crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum
        {
            let baseline = baseline?;
            amount = amount.add(
                &Expression::constant(
                    Interval {
                        lo: -baseline[1],
                        hi: -baseline[0],
                    },
                    1,
                ),
                &domains,
            )?;
        }
        let cost = amount.scale(objective_coefficient, &domains)?;
        reduced = reduced.add(&cost, &domains)?;
    }
    for entry in oracle
        .activity_rows
        .iter()
        .filter(|entry| entry.resource == resource)
    {
        if entry.coefficient == 0. {
            continue;
        }
        let dual = *duals.get(offset.checked_add(entry.row)?)?;
        let row_value = terminal
            .scale(entry.coefficient, &domains)?
            .scale(-dual, &domains)?;
        reduced = reduced.add(&row_value, &domains)?;
    }
    *lower = crate::core::rounding::add_down(*lower, reduced.range(&domains)?.lo)?;
    Some(())
}

/// Add a bounded terminal-arc lower bound and retain its refinement input.
/// Sink arcs with bit-identical value intervals have identical contributions
/// for this resource and need only one refinement entry. The bounded cache is
/// local to this resource's immutable pricing inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn add_bounded_terminal_contribution<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    resource: usize,
    normalization: crate::graph::coupled::PairedCostNormalization,
    objective_coefficient: f64,
    baseline: Option<[f64; 2]>,
    values: Interval,
    arc: usize,
    arc_lower: &mut f64,
    terminals: &mut Vec<PairedTerminalArcCharge>,
    terminal_contributions: &mut HashMap<(u64, u64), f64>,
    work: &mut usize,
    deadline: Instant,
) -> Option<()> {
    let key = (values.lo.to_bits(), values.hi.to_bits());
    let previous = terminals.last().filter(|terminal| {
        terminal.values.lo.to_bits() == values.lo.to_bits()
            && terminal.values.hi.to_bits() == values.hi.to_bits()
    });
    let prior_lower = previous
        .map(|terminal| terminal.lower)
        .or_else(|| terminal_contributions.get(&key).copied());
    let contribution_work = if prior_lower.is_some() {
        1
    } else {
        oracle.activity_rows.len().checked_add(1)?
    };
    *work = work.checked_sub(contribution_work)?;
    if Instant::now() >= deadline {
        return None;
    }
    let lower = match prior_lower {
        Some(lower) => lower,
        None => {
            let mut lower = 0.;
            add_paired_arc_contribution(
                oracle,
                offset,
                duals,
                resource,
                normalization,
                objective_coefficient,
                baseline,
                values,
                &mut lower,
            )?;
            if terminal_contributions.len() < 1024 {
                terminal_contributions.insert(key, lower);
            }
            lower
        }
    };
    *arc_lower = crate::core::rounding::add_down(*arc_lower, lower)?;
    if prior_lower.is_none() {
        terminals.push(PairedTerminalArcCharge {
            arc,
            values,
            floor: values.lo,
            lower,
        });
    }
    Some(())
}

/// Tighten a monotone CostBounded paired reduced-cost charge. Its dependent
/// value floor/ceiling applies to every continuation; the negative direction
/// also uses the remaining-hop growth envelope. Taking the minimum improvement
/// across feasible sink arcs keeps the correction valid when reachability is
/// unresolved.
pub(super) fn paired_prefix_terminal_correction<const D: usize>(
    graph: &crate::graph::Graph<D>,
    charges: &[PairedTerminalCharge],
    domains: &[PairedPrefixDomain],
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    remaining: usize,
    ceiling_cache: &mut HashMap<(usize, u64, usize), f64>,
    work: &mut usize,
    deadline: Instant,
) -> Option<f64> {
    let mut total = 0.;
    for (charge_index, charge) in charges.iter().enumerate() {
        // Root domains are emitted in definition order and every extension
        // preserves that order, so lookup need not rescan all earlier paired
        // resources for each terminal charge.
        let Ok(domain_index) =
            domains.binary_search_by_key(&charge.resource, |domain| domain.resource)
        else {
            continue;
        };
        let domain = &domains[domain_index];
        let terminal_ceiling = if charge.positive_direction {
            None
        } else {
            let key = (charge_index, domain.dependent_upper.to_bits(), remaining);
            let cached = (remaining >= 2)
                .then(|| ceiling_cache.get(&key).copied())
                .flatten();
            let ceiling = if let Some(cached) = cached {
                if Instant::now() >= deadline {
                    return None;
                }
                cached
            } else {
                let before = *work;
                let result = paired_terminal_ceiling_after_hops(
                    domain.dependent_upper,
                    charge.max_increment,
                    charge.max_waiting,
                    charge.max_lower,
                    charge.max_upper,
                    charge.upper_unbounded,
                    remaining,
                    work,
                    deadline,
                )?;
                // Keep memory bounded and avoid retaining one-step fixed
                // points, for which a lookup would save little work.
                if remaining >= 2 && before - *work >= 2 && ceiling_cache.len() < 4096 {
                    ceiling_cache.insert(key, result);
                }
                result
            };
            Some(ceiling)
        };
        let mut minimum = f64::INFINITY;
        for terminal in &charge.terminals {
            let contribution_work = oracle.activity_rows.len().checked_add(1)?;
            *work = work.checked_sub(contribution_work)?;
            if Instant::now() >= deadline || graph.arcs.get(terminal.arc)?.head != graph.sink {
                return None;
            }
            // This terminal arc cannot complete the current state when its
            // upper envelope lies below the prefix's dependent-value floor.
            if charge.lower_monotone && domain.dependent_lower > terminal.values.hi {
                continue;
            }
            let values = if charge.positive_direction {
                Interval {
                    lo: terminal.values.lo.max(domain.dependent_lower),
                    hi: terminal.values.hi,
                }
            } else {
                let terminal_upper = terminal_ceiling?.min(terminal.values.hi);
                if terminal_upper < terminal.floor {
                    continue;
                }
                Interval {
                    lo: terminal.values.lo,
                    hi: terminal_upper,
                }
            };
            let mut tightened = 0.;
            add_paired_arc_contribution(
                oracle,
                offset,
                duals,
                charge.resource,
                charge.normalization,
                charge.objective_coefficient,
                charge.baseline,
                values,
                &mut tightened,
            )?;
            let delta = crate::core::rounding::add_down(tightened, -terminal.lower)?;
            if delta < 0. {
                // Arithmetic did not prove a nonnegative improvement; skip
                // this resource's refinement while retaining the base bound.
                minimum = 0.;
                break;
            }
            minimum = minimum.min(delta);
        }
        if minimum.is_finite() {
            total = crate::core::rounding::add_down(total, minimum)?;
        }
    }
    Some(total)
}

/// Directed enclosure of every value on a prepared raw-slope function.
/// Each segment is affine, so its extrema occur at its endpoints; interval
/// arithmetic also encloses the function's sequential binary64 evaluation.
pub(super) fn slope_function_hull(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let pairs = function.pairs();
    let &(initial, _) = pairs.first()?;
    let mut value = Interval::point(initial);
    let mut lower = initial;
    let mut upper = initial;
    let mut previous = pairs.first()?.1;
    for &(slope, end) in pairs.iter().skip(1) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let span = end - previous;
        if !span.is_finite() || span <= 0. {
            return None;
        }
        let delta = Interval::point(slope).multiply(Interval::point(span))?;
        value = value.add(delta)?;
        lower = lower.min(value.lo);
        upper = upper.max(value.hi);
        previous = end;
    }
    Some(Interval {
        lo: lower,
        hi: upper,
    })
}

pub(super) fn slope_charge_hull(
    functions: &[crate::graph::coupled::PreparedSlopeFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = slope_function_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Directed hull for encoded state costs. Knot functions clamp interpolation
/// to their segment endpoints; slope functions retain their prepared
/// sequential arithmetic, which `slope_function_hull` encloses outward.
pub(super) fn encoded_charge_hull(
    functions: &[crate::graph::coupled::EncodedFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = match function {
            crate::graph::coupled::EncodedFunction::Knots { function } => {
                piecewise_charge_hull(std::slice::from_ref(function), work, deadline)?
            }
            crate::graph::coupled::EncodedFunction::SlopeSegments { function, .. } => {
                slope_function_hull(function, work, deadline)?
            }
        };
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Piecewise-linear source evaluation clamps interpolation to its segment's
/// endpoint values, so the global knot hull encloses every rounded charge.
pub(super) fn piecewise_charge_hull(
    functions: &[crate::graph::coupled::PiecewiseLinearFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let values = piecewise_value_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(values.lo);
        hull.hi = hull.hi.max(values.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

pub(super) fn piecewise_value_hull(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for point in &function.points {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline || !point[1].is_finite() {
            return None;
        }
        hull.lo = hull.lo.min(point[1]);
        hull.hi = hull.hi.max(point[1]);
    }
    (hull.lo.is_finite() && hull.hi.is_finite()).then_some(hull)
}

/// Sum a global per-arc charge range over all path lengths up to `hops`.
/// Zero is included at each position to relax routes that end earlier.
pub(super) fn repeated_charge_hull(
    per_arc: Interval,
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let term = Interval {
        lo: per_arc.lo.min(0.),
        hi: per_arc.hi.max(0.),
    };
    let mut total = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = total;
        total = total.add(term)?;
        // The term is invariant, so a bit-identical interval is a fixed
        // point of every remaining directed addition. This commonly skips
        // long cyclic walk caps when the per-arc charge is zero.
        if total.lo.to_bits() == previous.lo.to_bits()
            && total.hi.to_bits() == previous.hi.to_bits()
        {
            break;
        }
    }
    Some(total)
}

/// Outward lower enclosure of a cumulative/custom cost over every walk of at
/// most `hops` arcs. Increment and weight choices are relaxed independently,
/// while interval addition encloses the source program's per-arc rounding.
pub(super) fn accumulated_cost_hull(
    initial: f64,
    increments: &[f64],
    weights: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    if increments.len() != weights.len() {
        return None;
    }
    if hops == 0 {
        return Some(Interval::point(0.));
    }
    let mut increment_hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    let mut weight_hull = increment_hull;
    for (&increment, &weight) in increments.iter().zip(weights) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if !increment.is_finite() || !weight.is_finite() {
            return None;
        }
        increment_hull.lo = increment_hull.lo.min(increment);
        increment_hull.hi = increment_hull.hi.max(increment);
        weight_hull.lo = weight_hull.lo.min(weight);
        weight_hull.hi = weight_hull.hi.max(weight);
    }
    if increments.is_empty() {
        return Some(Interval::point(0.));
    }
    let mut state = Interval::point(initial);
    let mut cost = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous_state = state;
        let previous_cost = cost;
        state = state.add(increment_hull)?;
        let term = state.multiply(weight_hull)?;
        let term = Interval {
            lo: term.lo.min(0.),
            hi: term.hi.max(0.),
        };
        cost = cost.add(term)?;
        // Both the state and accumulated-cost intervals participate in
        // the next recurrence. Stop only if both are bit-identical, so no
        // future directed result or overflow can differ from this step.
        if state.lo.to_bits() == previous_state.lo.to_bits()
            && state.hi.to_bits() == previous_state.hi.to_bits()
            && cost.lo.to_bits() == previous_cost.lo.to_bits()
            && cost.hi.to_bits() == previous_cost.hi.to_bits()
        {
            break;
        }
    }
    Some(cost)
}

pub(super) fn suffix_table<const D: usize>(
    graph: &Graph<D>,
    arc_costs: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<SuffixTable> {
    hops.checked_add(1)?;
    // Validate the sink before allocating: an empty vertex set would make
    // the cell-count check zero even for an enormous requested row count.
    if graph.sink >= graph.vertices.len() {
        return None;
    }
    // Bound stored cells by the initial work budget, allocating rows as they
    // are computed. A stable row represents every later hop count, so a
    // large declared cap does not itself require a large allocation.
    let allocation_limit = *work;
    if graph.vertices.len() > allocation_limit {
        return None;
    }
    let mut suffix = vec![vec![None; graph.vertices.len()]];
    *suffix[0].get_mut(graph.sink)? = Some(0.);
    for remaining in 1..=hops {
        if Instant::now() >= deadline {
            return None;
        }
        if (remaining + 1).checked_mul(graph.vertices.len())? > allocation_limit {
            return None;
        }
        suffix.push(suffix.last()?.clone());
        let (before, current) = suffix.split_at_mut(remaining);
        let previous = &before[remaining - 1];
        let current = &mut current[0];
        for (index, arc) in graph.arcs.iter().enumerate() {
            if index % 256 == 0 && Instant::now() >= deadline {
                return None;
            }
            *work = work.checked_sub(1)?;
            let Some(tail) = previous.get(arc.head).copied().flatten() else {
                continue;
            };
            let candidate = crate::core::rounding::add_down(*arc_costs.get(index)?, tail)?;
            let value = current.get_mut(arc.tail)?;
            *value = Some(value.map_or(candidate, |old: f64| old.min(candidate)));
        }
        // The next row is a deterministic function of this row, with the
        // same arc order and directed arithmetic. Once consecutive rows are
        // bit-identical, every later row is identical too. This matters for
        // bounded cyclic walks whose relaxation stabilizes well before the
        // hop cap. Compare bits so signed zeros cannot change unnoticed.
        let fixed = remaining < hops
            && previous
                .iter()
                .zip(current.iter())
                .all(|(a, b)| match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                    _ => false,
                });
        if fixed {
            return Some(SuffixTable {
                rows: suffix,
                hops,
                stable: true,
            });
        }
    }
    Some(SuffixTable {
        rows: suffix,
        hops,
        stable: false,
    })
}

pub(super) fn physical_suffix_bounds<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    original_costs: Option<&OriginalVariableCosts>,
    work: &mut usize,
    deadline: Instant,
) -> Option<PhysicalSuffixBounds> {
    if !row_simple_supported(oracle) {
        return None;
    }
    if original_costs.is_none() {
        let allowance = *work / 8;
        let mut remaining = allowance;
        let exact = additive::prepare(oracle, offset, duals, &mut remaining, deadline);
        *work -= allowance - remaining;
        if exact.is_some() {
            return exact;
        }
    }
    let graph = &oracle.network.graph;
    if graph
        .coupled_kernel()?
        .definitions()
        .iter()
        .any(|definition| {
            !matches!(
                definition,
                CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
            ) && matches!(definition, CoupledResource::Plugin(_))
        })
    {
        // Plugin costs have no directed lower bound. Do not omit them from a
        // certificate.
        return None;
    }
    if oracle.network.arc_entries.len() != graph.arcs.len() {
        return None;
    }
    *work = work.checked_sub(
        graph
            .vertices
            .len()
            .checked_add(graph.arcs.len())?
            .checked_add(oracle.route_entries.len())?
            .checked_add(oracle.network.route_entries.len())?,
    )?;
    let dual = |row: usize| -> Option<f64> {
        let value = *duals.get(row)?;
        value.is_finite().then_some(value)
    };
    let mut route_constant = 0.;
    for &(row, coefficient) in &oracle.route_entries {
        route_constant = crate::core::rounding::add_down(
            route_constant,
            crate::core::rounding::multiply_down(-dual(row)?, coefficient)?,
        )?;
    }
    for &(row, coefficient) in oracle.network.route_entries.iter() {
        route_constant = crate::core::rounding::add_down(
            route_constant,
            crate::core::rounding::multiply_down(-dual(offset.checked_add(row)?)?, coefficient)?,
        )?;
    }
    let mut arc_costs = if let Some(original) = original_costs {
        original_arc_cost_lower_bounds(oracle, original, offset, duals, work, deadline)?
    } else {
        let mut costs = Vec::with_capacity(graph.arcs.len());
        for (index, arc) in graph.arcs.iter().enumerate() {
            let mut cost = arc.cost;
            for &(row, coefficient) in oracle.network.arc_entries.get(index)? {
                *work = work.checked_sub(1)?;
                cost = crate::core::rounding::add_down(
                    cost,
                    crate::core::rounding::multiply_down(
                        -dual(offset.checked_add(row)?)?,
                        coefficient,
                    )?,
                )?;
            }
            costs.push(cost);
        }
        costs
    };
    let (other_coupled_lower, paired_arc_lower, paired_terminal_charges) =
        coupled_lower_bounds(oracle, offset, duals, work, deadline)?;
    let hops = graph.original_max_arcs();
    // A paired objective or activity is charged once at the terminal state.
    // Its directed per-arc envelope can therefore be attached to the final
    // physical arc entering the sink. The ordinary suffix DP then keeps the
    // terminal arc and physical path costs correlated without storing a
    // separate resource-state table.
    for (index, arc) in graph.arcs.iter().enumerate() {
        if arc.head == graph.sink {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline {
                return None;
            }
            arc_costs[index] =
                crate::core::rounding::add_down(arc_costs[index], paired_arc_lower[index])?;
        }
    }
    let suffix = suffix_table(graph, &arc_costs, hops, work, deadline)?;
    Some(PhysicalSuffixBounds {
        suffix,
        arc_costs,
        route_constant,
        other_coupled_lower,
        paired_terminal_charges,
        exact_additive: false,
    })
}

pub(super) fn original_arc_cost_lower_bounds<const D: usize>(
    oracle: &Oracle<D>,
    original: &OriginalVariableCosts,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<f64>> {
    let graph = &oracle.network.graph;
    let index: HashMap<_, _> = graph
        .arcs
        .iter()
        .enumerate()
        .map(|(position, arc)| (arc.id, position))
        .collect();
    let row_cost = |mut value: Interval, entries: &[(usize, f64)], work: &mut usize| {
        for &(row, coefficient) in entries {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline || !coefficient.is_finite() {
                return None;
            }
            let dual = *duals.get(offset.checked_add(row)?)?;
            value = value.add(Interval::point(-dual).multiply(Interval::point(coefficient))?)?;
        }
        Some(value)
    };
    let mut costs = Vec::with_capacity(graph.arcs.len());
    for (position, arc) in graph.arcs.iter().enumerate() {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let saved = original.arcs.get(&arc.id)?;
        let entries = oracle.network.arc_entries.get(position)?;
        if !entries.starts_with(&saved.compiled_entries) {
            return None;
        }
        let compiled_delta = Interval::point(arc.cost).add(Interval::point(-saved.compiled))?;
        let base = Interval::point(saved.base).add(compiled_delta)?;
        let base = row_cost(base, &saved.entries, work)?;
        let additional = entries.get(saved.compiled_entries.len()..)?;
        costs.push(row_cost(base, additional, work)?);
    }
    for variable in &original.variables {
        if variable.accumulated_resource.is_some() {
            continue;
        }
        let reduced = row_cost(Interval::point(variable.cost), &variable.entries, work)?;
        let scaled = reduced.scale(1e8)?.add(Interval::point(0.5))?;
        if !scaled.lo.is_finite()
            || !scaled.hi.is_finite()
            || scaled.lo.abs() > 9_007_199_254_740_992.
            || scaled.hi.abs() > 9_007_199_254_740_992.
        {
            return None;
        }
        let rounded = Interval {
            lo: scaled.lo.floor(),
            hi: scaled.hi.floor(),
        }
        .scale(1e-8)?;
        for &(arc, association) in &variable.arcs {
            *work = work.checked_sub(1)?;
            if Instant::now() >= deadline || !association.is_finite() {
                return None;
            }
            let position = *index.get(&arc)?;
            let cost = costs.get_mut(position)?;
            *cost = cost.add(rounded.scale(association)?)?;
        }
    }
    Some(costs.into_iter().map(|cost| cost.lo).collect())
}
