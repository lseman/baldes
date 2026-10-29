// Route-local pricing, physical minima, and column bounds.
use super::*;
pub(super) fn path<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Option<PathPricing>> {
    let graph = &oracle.network.graph;
    let mut scalar = resource_groups::domain_graph(oracle).ok()?.into_owned();
    scalar.set_coupled_resources(&[]).ok()?;
    let problem = crate::master::routing::RoutingContext::set_partitioning(scalar, 1.).ok()?;
    let ids: Vec<_> = arcs.iter().map(|&p| graph.arcs[p].id).collect();
    let skeleton = Route {
        nodes: std::iter::once(graph.source as i32)
            .chain(arcs.iter().map(|&p| graph.arcs[p].head as i32))
            .collect(),
        arcs: ids.clone(),
        cost: 0.,
        original_cost: 0.,
        resources: vec![],
        resource_profile: vec![],
        coupled_values: vec![],
        coupled_controls: vec![],
    };
    if crate::master::routing::exact_scalar_path_feasibility(&problem, &skeleton) == Some(false) {
        return Some(None);
    }
    let route = problem.elementary_route(&ids);
    let kernel = graph.coupled_kernel()?;
    let resources = resources(oracle);
    let mut domains = vec![];
    let mut minimum = vec![];
    let mut maximum = vec![];
    for &r in &resources {
        let Some(domain) = domain(oracle, arcs, r, work, iterations, deadline)? else {
            return Some(None);
        };
        domains.push(Interval {
            lo: domain.lower,
            hi: domain.upper,
        });
        minimum.push(domain.minimum);
        maximum.push(domain.maximum);
    }
    // Match ordinary replay: numeric/front prefix, slope pairs, then knot pairs.
    // Every paired charge is removed from the prefix before being inserted at
    // its own position; witness metadata must not change parenthesization.
    let mut paired_order: Vec<_> = kernel
        .definitions()
        .iter()
        .enumerate()
        .filter_map(|(r, definition)| match definition {
            CoupledResource::SlopePaired { mode, .. }
            | CoupledResource::PairedPiecewiseLinear { mode, .. }
                if *mode != crate::graph::coupled::PairedResourceMode::Feasibility =>
            {
                Some(r)
            }
            _ => None,
        })
        .collect();
    paired_order.sort_by_key(|&r| {
        (
            !matches!(kernel.definitions()[r], CoupledResource::SlopePaired { .. }),
            r,
        )
    });
    let mut base = graph.as_ref().clone();
    let weights: Vec<_> = kernel
        .definitions()
        .iter()
        .enumerate()
        .filter_map(|(r, _)| {
            let q = kernel.objective_coefficient(r)?;
            (paired_order.contains(&r) || q != 1.)
                .then_some((r, if paired_order.contains(&r) { 0. } else { q }))
        })
        .collect();
    base.set_coupled_objective_coefficients(&weights).ok()?;
    let controls = sorted(minimum.clone());
    let source = SourceColumn {
        kernel: base.coupled_kernel()?,
        controls: &controls,
        variable_resources: &resources,
    };
    let terms = components(oracle, offset, duals, arcs, work, deadline, source)?;
    let Some(mut terms) = terms else {
        return Some(None);
    };
    let numeric = base
        .coupled_kernel()?
        .replay_with_controls_using(arcs, &controls, work, Some(deadline))
        .ok()??;
    terms.coupled = hull(terms.coupled, numeric.cost)?;
    // Enclose both source coefficients and the actual registered constant terms.
    if let Some(route) = route.as_ref() {
        for (r, value) in entries_base(oracle, route, offset).ok()? {
            let old = terms.rows.entry(r).or_insert(Interval::point(0.));
            *old = hull(*old, value)?;
        }
    }
    let n = resources.len();
    let numeric_components = if paired_order.iter().any(|r| !resources.contains(r)) {
        *work = work.checked_sub(
            arcs.len()
                .checked_mul(kernel.definitions().len().checked_add(1)?)?,
        )?;
        Some(kernel.replay_cost_components(arcs).ok()??)
    } else {
        None
    };
    let mut coupled = Expression::constant(terms.coupled, n);
    for r in paired_order {
        let charge = if let Some(i) = resources.iter().position(|&resource| resource == r) {
            let coefficient = kernel.effective_objective_coefficient(r)?;
            let mut value = Expression::variable(i, n);
            if coefficient != 0.
                && matches!(
                    kernel.definitions()[r],
                    CoupledResource::SlopePaired {
                        normalization:
                            crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
                        ..
                    } | CoupledResource::PairedPiecewiseLinear {
                        normalization:
                            crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
                        ..
                    }
                )
            {
                let baseline = kernel.paired_initial_minimum(r, work, Some(deadline))?;
                value = value.add(
                    &Expression::constant(
                        Interval {
                            lo: -baseline[1],
                            hi: -baseline[0],
                        },
                        n,
                    ),
                    &domains,
                )?;
            }
            value.scale(coefficient, &domains)?
        } else {
            let (valid, range) =
                kernel.certify_paired_objective_component(arcs, r, work, Some(deadline))?;
            if !valid {
                return Some(None);
            }
            Expression::constant(
                hull(
                    Interval {
                        lo: range[0],
                        hi: range[1],
                    },
                    *numeric_components.as_ref()?.resources.get(r)?,
                )?,
                n,
            )
        };
        coupled = coupled.add(&charge, &domains)?;
    }
    let mut reduced = Expression::constant(terms.physical, n).add(&coupled, &domains)?;
    let mut rows: BTreeMap<_, _> = terms
        .rows
        .into_iter()
        .map(|(r, value)| (r, Expression::constant(value, n)))
        .collect();
    let mut paired = BTreeMap::<usize, Expression>::new();
    for c in &oracle.activity_rows {
        if c.coefficient == 0. {
            continue;
        }
        let Some(i) = resources.iter().position(|&r| r == c.resource) else {
            continue;
        };
        let value = Expression::variable(i, n).scale(c.coefficient, &domains)?;
        let row = paired
            .entry(offset + c.row)
            .or_insert_with(|| Expression::constant(Interval::point(0.), n));
        *row = row.add(&value, &domains)?;
    }
    for (r, value) in paired {
        let row = rows
            .entry(r)
            .or_insert_with(|| Expression::constant(Interval::point(0.), n));
        *row = row.add(&value, &domains)?;
    }
    for (r, value) in rows {
        reduced = reduced.add(&value.scale(-*duals.get(r)?, &domains)?, &domains)?;
    }
    let lower = source_control_lower_bound(&domains, &reduced)?;
    let greedy = resources
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if reduced.coefficient(i).hi < 0. {
                maximum[i].clone()
            } else {
                minimum[i].clone()
            }
        })
        .collect();
    let sign_ambiguous = (0..resources.len())
        .filter(|&index| sign_ambiguous(reduced.coefficient(index)))
        .collect();
    Some(Some(PathPricing {
        lower,
        route,
        resources,
        domains,
        minimum,
        maximum,
        greedy,
        sign_ambiguous,
    }))
}

pub(in crate::master::model) fn minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let allowance = *work / 8;
    let mut remaining = allowance;
    let exact = additive::attained_minimum(oracle, offset, duals, &mut remaining, deadline);
    *work -= allowance - remaining;
    if exact.is_some() {
        return exact;
    }
    // Retain the polynomial relaxation as a fallback, but refine it when
    // complete source coverage fits the budget. An independent terminal
    // envelope can remain strictly below every feasible route and otherwise
    // prevent the master from certifying convergence after pricing closes.
    let allowance = *work / 4;
    let relaxed = if allowance != 0 {
        let mut remaining = allowance;
        let relaxed = physical_minimum(oracle, offset, duals, &mut remaining, deadline);
        *work -= allowance - remaining;
        relaxed
    } else {
        None
    };
    if relaxed == Some(None) {
        return relaxed;
    }
    let mut iterations = 100_000;
    let mut lower: Option<f64> = None;
    let complete = bounds::columns::visit_physical_paths(oracle, work, deadline, |arcs, work| {
        let Some(result) = path(oracle, offset, duals, arcs, work, &mut iterations, deadline)
        else {
            return Err(crate::enumeration::PathDomainFailure::Arithmetic);
        };
        if let Some(result) = result {
            lower = Some(lower.map_or(result.lower, |old| old.min(result.lower)));
        }
        Ok(())
    });
    if Instant::now() >= deadline {
        return None;
    }
    if complete.is_some() {
        Some(lower)
    } else {
        relaxed
    }
}

pub(super) fn physical_minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let graph = &oracle.network.graph;
    // Empty routes have their own objective/activity normalization. Keep their
    // existing source-control proof rather than applying a nonempty-route
    // constant to the zero-hop suffix row.
    if graph.source == graph.sink {
        return None;
    }
    let bounds = physical_suffix_bounds(oracle, offset, duals, None, work, deadline)?;
    let hops = bounds.suffix.len().checked_sub(1)?;
    let tail = bounds.suffix.get(hops)?.get(graph.source).copied()?;
    if Instant::now() >= deadline {
        return None;
    }
    let Some(tail) = tail else {
        // The complete bounded-walk superset cannot reach the sink.
        return Some(None);
    };
    let constant =
        crate::core::rounding::add_down(bounds.route_constant, bounds.other_coupled_lower)?;
    let lower = crate::core::rounding::add_down(tail, constant)?;
    lower.is_finite().then_some(Some(lower))
}

pub(in crate::master::model) fn column_bound<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    column_bound_using(oracle, offset, duals, arcs, work, &mut 100_000, deadline)
}
pub(in crate::master::model) fn column_bound_using<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
    arcs: &[usize],
    work: &mut usize,
    iterations: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    Some(path(oracle, offset, duals, arcs, work, iterations, deadline)?.map(|p| p.lower))
}

pub(super) fn process_path<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    duals: &[f64],
    arcs: &[usize],
    deadline: Instant,
    work: &mut usize,
    iterations: &mut usize,
    minimum: &mut f64,
    routes: &mut Vec<Route>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    let result = path(
        oracle,
        state.shared_offset,
        duals,
        arcs,
        work,
        iterations,
        deadline,
    )
    .ok_or_else(|| incomplete("paired source-control pricing control coverage incomplete"))?;
    let Some(result) = result else {
        return Ok(());
    };
    *minimum = minimum.min(result.lower);
    let Some(ref route) = result.route else {
        return Ok(());
    };
    // Row boundary proposals enrich the restricted master. They are never
    // used to assert exhaustive control coverage or source infeasibility.
    let initial = route
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(result.greedy.clone()))?;
    let coefficients: BTreeMap<_, _> = entries(oracle, &initial, state.shared_offset)?
        .into_iter()
        .collect();
    let mut quantities = vec![1.];
    for count in [
        oracle.network.multiplicity.lower.ceil(),
        oracle.multiplicity_upper().floor(),
    ] {
        if count.is_finite() && (1. ..=9_007_199_254_740_992.).contains(&count) {
            quantities.push(count);
        }
    }
    quantities.sort_by(f64::total_cmp);
    quantities.dedup();
    let mut choices = vec![
        result.minimum.clone(),
        result.maximum.clone(),
        result.greedy.clone(),
    ];
    let combinations = if result.sign_ambiguous.is_empty() {
        0
    } else {
        let shift = u32::try_from(result.sign_ambiguous.len())
            .map_err(|_| incomplete("paired endpoint proposal dimension overflow"))?;
        let count = 1usize
            .checked_shl(shift)
            .ok_or_else(|| incomplete("paired endpoint proposal dimension overflow"))?;
        *work = work
            .checked_sub(count)
            .ok_or_else(|| incomplete("paired endpoint proposal work budget exhausted"))?;
        count
    };
    for quantity in quantities {
        if let Some(controls) = joint_controls(
            state,
            oracle,
            arcs,
            &result,
            &initial,
            quantity,
            (work, iterations),
            deadline,
        ) {
            choices.extend(controls);
        }
    }
    for (i, &resource) in result.resources.iter().enumerate() {
        for c in oracle
            .activity_rows
            .iter()
            .filter(|c| c.resource == resource && c.coefficient != 0.)
        {
            let r = state.shared_offset + c.row;
            let coefficient: f64 = oracle
                .activity_rows
                .iter()
                .filter(|other| other.resource == resource && other.row == c.row)
                .map(|other| other.coefficient)
                .sum();
            let value = initial
                .coupled_values
                .iter()
                .find(|(r, _)| *r == resource)
                .map(|(_, v)| *v)
                .unwrap_or(0.);
            let target =
                value + (state.rhs[r] - coefficients.get(&r).copied().unwrap_or(0.)) / coefficient;
            if !target.is_finite() || target < result.domains[i].lo || target > result.domains[i].hi
            {
                continue;
            }
            if let Some(proposal) =
                target_control(oracle, arcs, resource, target, work, iterations, deadline)
            {
                if let Some(control) = proposal.control {
                    let mut selection = result.greedy.clone();
                    selection[i] = control;
                    choices.push(selection);
                }
            }
        }
    }
    // Stream the exponential endpoint sweep: the work budget bounds its
    // count, while the deadline also bounds cloning and source replay.
    let endpoints = (0..combinations).map(|mask| {
        let mut controls = result.greedy.clone();
        for (bit, &index) in result.sign_ambiguous.iter().enumerate() {
            controls[index] = if mask & (1usize << bit) == 0 {
                result.minimum[index].clone()
            } else {
                result.maximum[index].clone()
            };
        }
        controls
    });
    for controls in choices.into_iter().chain(endpoints) {
        if Instant::now() >= deadline {
            return Err(incomplete("paired endpoint proposal deadline exhausted"));
        }
        let candidate = route
            .clone()
            .with_paired_controls(&oracle.network.graph, sorted(controls))?;
        if lifecycle::integer_route_feasible(oracle, &candidate)?
            && seen.insert(candidate.identity())
        {
            routes.push(candidate);
        }
    }
    Ok(())
}
