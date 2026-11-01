//! Per-network pricing preparation, route collection and bound admission.
use super::*;

impl<const D: usize> State<D> {
    pub(super) fn price(&mut self, phase: usize, duals: &[f64]) -> Result<cg::Answer> {
        for &(network, resource, bounds) in &self.preprocessing.paired_owner_bounds {
            if self
                .oracles
                .iter()
                .find(|oracle| oracle.network.id == network)
                .is_some_and(|oracle| oracle.multiplicity_upper() == 0.)
            {
                continue;
            }
            if crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported([
                bounds.lower,
                bounds.upper,
            ]) {
                return Err(invalid(format!("non-zero lower bounds on variables associated to arcs are not supported (network {network}, paired resource {resource}, propagated owner bounds [{}, {}])", bounds.lower, bounds.upper)));
            }
        }
        // Explicit registration bounds remain relevant when propagation was
        // disabled or stopped before it could publish an owner coordinate.
        for oracle in &self.oracles {
            if oracle.multiplicity_upper() == 0. {
                continue;
            }
            if let Some(kernel) = oracle.network.graph.coupled_kernel() {
                for &(resource, bounds) in kernel.paired_owner_registration_bounds() {
                    if self
                        .preprocessing
                        .paired_owner_bounds
                        .iter()
                        .any(|&(network, index, _)| {
                            network == oracle.network.id && index == resource
                        })
                    {
                        continue;
                    }
                    if crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported(
                        bounds,
                    ) {
                        return Err(invalid(format!("non-zero lower bounds on variables associated to arcs are not supported (network {}, paired resource {resource}, input owner bounds {:?})", oracle.network.id, bounds)));
                    }
                }
            }
        }
        self.correction = None;
        self.gradient = None;
        self.scheduler.complete = false;
        let mut projected = std::mem::take(&mut self.pricing_duals);
        let duals = if self.retired_rows.is_empty() {
            duals
        } else {
            projected.clear();
            projected.extend_from_slice(duals);
            for &row in &self.retired_rows {
                if let Some(value) = projected.get_mut(row) {
                    *value = 0.;
                }
            }
            &projected
        };
        let mut rhs = std::mem::take(&mut self.pricing_rhs);
        rhs.clone_from(&self.rhs);
        for &row in &self.retired_rows {
            rhs[row] = 0.;
        }
        let context = crate::master::pricing::Context {
            phase,
            duals,
            rhs: &rhs,
            deadline: Some(self.deadline),
            admission: self.box_pricing_admission,
            master: self.pricing_master,
            gap: self.pricing_gap,
        };
        let tolerance = self.scheduler.tolerance;
        let fresh = self.scheduler.fresh();
        let mut scheduler = std::mem::replace(&mut self.scheduler, fresh);
        let result = crate::master::pricing::price(
            &mut NetworkPricingAdapter {
                state: self,
                tolerance,
            },
            &mut scheduler,
            context,
        );
        self.scheduler = scheduler;
        let unpriced = result.is_ok() && self.has_unpriced_paired_terms(duals);
        self.pricing_duals = projected;
        self.pricing_rhs = rhs;
        let mut batch = result?;
        if unpriced {
            // The raw operator completed, but its unit-charge minimum is not
            // a lower bound for the associated owner expression. Keep the raw
            // completion signal (also used to request explicit enumeration)
            // without publishing a numerical or certified owner bound.
            batch.correction = None;
            batch.gradient = None;
        }
        self.correction = batch.correction;
        self.gradient = batch.gradient;
        Ok(batch.answer)
    }
}

/// Adapter calls receive the round's already-projected physical duals.
pub(super) struct NetworkPricingAdapter<'a, const D: usize> {
    pub(super) state: &'a mut State<D>,
    pub(super) tolerance: f64,
}
impl<const D: usize> crate::master::pricing::Networks<D> for NetworkPricingAdapter<'_, D> {
    fn active(&self) -> Vec<usize> {
        self.state
            .oracles
            .iter()
            .enumerate()
            .filter_map(|(p, o)| (o.multiplicity_upper() != 0.).then_some(p))
            .collect()
    }
    fn prepare(
        &mut self,
        index: usize,
        phase: usize,
        duals: &[f64],
        admission: bool,
    ) -> Result<crate::master::pricing::Request> {
        use crate::master::pricing::{Envelopes, Request};
        let oracle = &mut self.state.oracles[index];
        if self.state.universe.is_some()
            || self.state.gap_pool_cutoff.is_some()
            || oracle.solver.enumeration.is_some()
        {
            return Ok(Request {
                phase,
                exact: true,
                offset: 0.,
                lower: oracle.network.multiplicity.lower,
                upper: oracle.multiplicity_upper(),
                envelopes: Envelopes::default(),
            });
        }
        oracle.costs.clear();
        oracle.costs.extend(
            oracle
                .network
                .graph
                .arcs
                .iter()
                .zip(oracle.network.arc_entries.iter())
                .map(|(arc, coefficients)| {
                    coefficients.iter().fold(arc.cost, |cost, &(row, value)| {
                        cost - duals[self.state.shared_offset + row] * value
                    })
                }),
        );
        if !oracle.activity_rows.is_empty() {
            super::activity_rows::price(oracle, duals, self.state.shared_offset)?;
        }
        for &(cut, row) in &oracle.nonlinear_rows {
            oracle.solver.cuts.cuts[cut].dual = duals[self.state.shared_offset + row];
        }
        if !oracle.nonlinear_rows.is_empty() {
            // Modern rank-one retrieval filters and orders raw prices before
            // rounding. Keep k-path rows in registry order, as the routing
            // adapter does. Exact pricing uses separate raw cut inputs.
            let registry = &mut oracle.solver.cuts;
            let mut order = std::mem::take(&mut registry.pricing_order);
            order.clear();
            order.extend((0..registry.cuts.len()).filter(|&index| {
                !registry.retired.contains(&index)
                    && !registry.cuts[index].visit_once
                    && registry.cuts[index].dual.abs() > 1e-6
            }));
            crate::cuts::shared::stdsort::stable_sort_with_buffer(
                &mut order,
                &mut self.state.pricing_order_scratch,
                |&a, &b| {
                    let (va, vb) = (-registry.cuts[a].dual, -registry.cuts[b].dual);
                    va > vb + 1e-6 || (!(vb - 1e-6 > va) && a < b)
                },
            );
            registry.pricing_order = order;
            registry.round_duals = true;
        }
        for &(cut, row) in &oracle.knapsack_rows {
            oracle.solver.cuts.knapsack_cuts[cut].dual = duals[self.state.shared_offset + row];
        }
        if oracle.presence_rows.iter().any(|&(input, row)| {
            oracle.solver.presence_rows()[input].dual != duals[self.state.shared_offset + row]
        }) {
            let mut prices: Vec<_> = oracle
                .solver
                .presence_rows()
                .iter()
                .map(|r| r.dual)
                .collect();
            for &(input, row) in &oracle.presence_rows {
                prices[input] = duals[self.state.shared_offset + row];
            }
            oracle.solver.set_presence_duals(&prices)?;
        }
        let cached = self
            .state
            .original_variable_costs
            .iter()
            .find(|model| model.network == oracle.network.id)
            .and_then(|model| {
                model.cached(
                    &oracle.network,
                    &oracle.index,
                    duals,
                    self.state.shared_offset,
                )
            });
        if let Some(cached) = cached {
            oracle.solver.cache_physical_costs(&oracle.costs, &cached)?;
        } else {
            oracle.solver.cache_variable_costs(&oracle.costs)?;
        }
        let offset = oracle
            .route_entries
            .iter()
            .map(|&(row, value)| -duals[row] * value)
            .sum::<f64>()
            - oracle
                .network
                .route_entries
                .iter()
                .map(|&(row, value)| duals[self.state.shared_offset + row] * value)
                .sum::<f64>();

        let original_box = self.state.grouped_variable_boxes.get(&oracle.network.id);
        let envelopes = if admission {
            let arc_box = self.state.variable_boxes.get(&oracle.network.id);
            let finite = if let Some(groups) = original_box {
                grouped_variable_box(
                    oracle,
                    offset,
                    groups,
                    arc_box.map(Vec::as_slice),
                    duals,
                    self.state.shared_offset,
                )
            } else {
                arc_box.map_or_else(
                    || required_nonnegative_box(oracle, offset),
                    |bounds| finite_variable_box(oracle, offset, bounds),
                )
            };
            Envelopes {
                nonnegative: nonnegative_box(oracle, offset),
                finite,
                near_zero: near_zero_box(oracle, offset, self.tolerance).map(
                    |(minimum, correction)| BoxAdmission {
                        minimum,
                        correction,
                        zero_quantity: minimum >= 0.,
                    },
                ),
                original: original_box.is_some(),
            }
        } else {
            Envelopes::default()
        };
        // Pricing omits the route-constant dual contribution from arc costs.
        // Its zero threshold therefore has the opposite sign to that offset.
        oracle.solver.controls.zero_threshold = -offset;
        Ok(Request {
            phase,
            exact: oracle.solver.phases[phase].exact(),
            offset,
            lower: oracle.network.multiplicity.lower,
            upper: oracle.multiplicity_upper(),
            envelopes,
        })
    }
    fn enter(&mut self, index: usize) -> &mut Solver<D> {
        let oracle = &mut self.state.oracles[index];
        oracle.calls += 1;
        &mut oracle.solver
    }
    fn retained(
        &mut self,
        index: usize,
        duals: &[f64],
    ) -> Option<Result<crate::master::pricing::Retained>> {
        let gap_universe;
        let universe = if let Some(universe) = self.state.universe.as_ref() {
            universe
        } else if self.state.gap_pool_cutoff.is_some()
            || self.state.oracles[index].solver.enumeration.is_some()
        {
            let oracle = &self.state.oracles[index];
            gap_universe = self
                .state
                .routes
                .iter()
                .enumerate()
                .filter(|(_, route)| {
                    route.network == oracle.network.id && oracle.is_enumerated(&route.route)
                })
                .map(|(tag, _)| tag)
                .collect::<Vec<_>>();
            &gap_universe
        } else {
            return None;
        };
        Some((|| {
            let oracle = &mut self.state.oracles[index];
            oracle.calls += 1;
            let mut columns = vec![];
            for &tag in universe {
                let route = &self.state.routes[tag];
                if route.network != oracle.network.id
                    || route.route.arcs.iter().any(|id| {
                        oracle
                            .proved_zero_arcs
                            .get(oracle.index[id])
                            .copied()
                            .unwrap_or(false)
                    })
                {
                    continue;
                }
                let cap = self
                    .state
                    .route_upper_bounds
                    .get(&tag)
                    .copied()
                    .unwrap_or(oracle.multiplicity_upper())
                    .min(oracle.multiplicity_upper());
                let mut coefficients = entries(oracle, &route.route, self.state.shared_offset)?;
                coefficients.retain(|(row, _)| !self.state.retired_rows.contains(row));
                let reduced = coefficients
                    .iter()
                    .fold(route.route.original_cost, |v, &(r, a)| v - duals[r] * a);
                columns.push((
                    cg::Column {
                        cost: route.route.original_cost,
                        reduced_cost: reduced,
                        entries: coefficients,
                        acceptable: true,
                        tag: tag as i32,
                        identity: format!("{}:{}", route.network, route.route.identity()),
                    },
                    cap,
                ));
            }
            Ok(crate::master::pricing::Retained {
                minimum_bound: None,
                columns,
                independent_caps: true,
                maximum: usize::MAX,
            })
        })())
    }
    fn collect(
        &mut self,
        index: usize,
        routes: Vec<Route>,
        offset: f64,
        duals: &[f64],
    ) -> Result<crate::master::pricing::Collected> {
        let oracle = &mut self.state.oracles[index];
        let mut columns = Vec::with_capacity(routes.len());
        let mut new_improving_columns = 0;
        let mut best: Option<(f64, Vec<(usize, f64)>, Route)> = None;
        for mut route in routes {
            let mut coefficients = entries(oracle, &route, self.state.shared_offset)?;
            coefficients.retain(|(row, _)| !self.state.retired_rows.contains(row));
            let cost = original_costs::route_cost(oracle, &route)?;
            let reduced_cost = coefficients
                .iter()
                .fold(cost, |value, &(row, coefficient)| {
                    value - duals[row] * coefficient
                });
            if !cost.is_finite() || !reduced_cost.is_finite() {
                return Err(invalid("network route cost overflow"));
            }
            if best
                .as_ref()
                .map_or(true, |(value, _, _)| reduced_cost < *value)
            {
                best = Some((reduced_cost, coefficients.clone(), route.clone()));
            }
            route.original_cost = cost;
            route.cost += offset;
            let identity = route.identity();
            let key = (oracle.network.id, identity.clone());
            let tag = if let Some(&tag) = self.state.tags.get(&key) {
                tag
            } else {
                new_improving_columns += usize::from(reduced_cost < -self.tolerance);
                let tag = i32::try_from(self.state.routes.len())
                    .map_err(|_| invalid("network route tag overflow"))?;
                Arc::make_mut(&mut self.state.tags).insert(key, tag);
                Arc::make_mut(&mut self.state.routes).push(Arc::new(NetworkRoute {
                    network: oracle.network.id,
                    route,
                }));
                tag
            };
            columns.push(cg::Column {
                cost,
                reduced_cost,
                entries: coefficients,
                acceptable: true,
                tag,
                identity: format!("{}:{identity}", oracle.network.id),
            });
        }

        let (best_route, best_entries) = best
            .map(|(_, entries, route)| (Some(route), Some(entries)))
            .unwrap_or_default();
        Ok(crate::master::pricing::Collected {
            columns,
            fresh: new_improving_columns,
            best_route,
            best_entries,
        })
    }
    fn static_contribution(&self, duals: &[f64]) -> Option<(f64, Vec<(usize, f64)>)> {
        master_variables::correction(self.state, duals)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/native_contribution_tests.rs"]
mod native_contribution_tests;

/// Nonnegative orthant case of the native variable-bound pricing test.
/// No unit arc cap is assumed: non-elementary walks can repeat an arc.
fn additive<const D: usize>(oracle: &Oracle<D>) -> crate::master::pricing::Additive<'_, D> {
    crate::master::pricing::Additive {
        graph: &oracle.network.graph,
        costs: &oracle.costs,
        lower: oracle.network.multiplicity.lower,
        upper: oracle.multiplicity_upper(),
        supported: oracle.nonlinear_rows.is_empty()
            && oracle.knapsack_rows.is_empty()
            && oracle.presence_rows.is_empty()
            && oracle.resource_groups.is_empty()
            && oracle.ryan_foster_rows.is_empty()
            && oracle.activity_rows.is_empty(),
    }
}
fn nonnegative_box<const D: usize>(oracle: &Oracle<D>, offset: f64) -> bool {
    additive(oracle).nonnegative(offset)
}
type BoxAdmission = crate::master::pricing::Estimate;

fn finite_variable_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    bounds: &[crate::master::preprocessing::Bounds],
) -> Option<BoxAdmission> {
    if !offset.is_finite()
        || bounds.len() != oracle.costs.len()
        || oracle.costs.iter().any(|cost| !cost.is_finite())
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
        || !oracle.ryan_foster_rows.is_empty()
        || !oracle.activity_rows.is_empty()
        || oracle.network.graph.has_coupled_resources()
    {
        return None;
    }
    let mut negative = crate::master::numerics::ExactSum::default();
    for (arc, (&cost, bounds)) in oracle.costs.iter().zip(bounds).enumerate() {
        if cost == 0. || oracle.solver.forbidden_arcs.get(arc) == Some(&true) {
            continue;
        }
        let selected = if cost < 0. {
            bounds.upper
        } else {
            bounds.lower
        };
        if !selected.is_finite() {
            return None;
        }
        negative.product_upper(-cost, selected)?;
    }
    negative.add(-offset)?;
    // Preserve negative remainders even when naive cancellation rounds them away.
    let lower = -negative.upper()?;
    box_correction(oracle, lower)
}

fn grouped_variable_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    groups: &grouped_boxes::PreparedBox,
    arc_bounds: Option<&[crate::master::preprocessing::Bounds]>,
    duals: &[f64],
    shared_offset: usize,
) -> Option<BoxAdmission> {
    if oracle.network.graph.has_coupled_resources() {
        return grouped_boxes::coupled_admission(oracle, groups, duals, shared_offset);
    }
    if !offset.is_finite()
        || oracle.costs.len() != groups.caps.len()
        || groups
            .variables
            .iter()
            .any(|v| v.accumulated_resource.is_some())
        || oracle.costs.iter().any(|c| !c.is_finite())
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
        || !oracle.ryan_foster_rows.is_empty()
        || !oracle.activity_rows.is_empty()
        || oracle.network.graph.has_coupled_resources()
    {
        return None;
    }
    let mut residual: Vec<_> = oracle
        .costs
        .iter()
        .map(|&cost| {
            let mut value = crate::master::numerics::ExactSum::default();
            value.add(-cost)?;
            Some(value)
        })
        .collect::<Option<_>>()?;
    let mut negative = crate::master::numerics::ExactSum::default();
    for variable in &groups.variables {
        let mut reduced = variable.cost;
        for &(row, a) in &variable.entries {
            reduced -= *duals.get(shared_offset.checked_add(row)?)? * a;
        }
        if !reduced.is_finite() {
            return None;
        }
        if reduced != 0. {
            let endpoint = if reduced < 0. {
                variable.bounds.upper
            } else {
                variable.bounds.lower
            };
            if !endpoint.is_finite() {
                return None;
            }
            negative.product_upper(-reduced, endpoint)?;
        }
        for &(arc, a) in &variable.arcs {
            residual[arc].product_upper(reduced, a)?;
        }
    }
    for (arc, value) in residual.iter().enumerate() {
        if oracle.solver.forbidden_arcs.get(arc) == Some(&true) {
            continue;
        }
        let cost = -value.upper()?;
        if cost == 0. {
            continue;
        }
        let lower = arc_bounds.map_or(0., |b| b[arc].lower.max(0.));
        let upper = arc_bounds.map_or(groups.caps[arc], |b| b[arc].upper.min(groups.caps[arc]));
        let endpoint = if cost < 0. { upper } else { lower };
        if !endpoint.is_finite() {
            return None;
        }
        negative.product_upper(-cost, endpoint)?;
    }
    negative.add(-offset)?;
    box_correction(oracle, -negative.upper()?)
}

/// Native's implicit [0,infinity] variable box gives a lower estimate,
/// including the route offset and required quantity. A first feasible route cannot be replaced
/// by zero selection; every such estimate goes through the master-gap gate.
fn required_nonnegative_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
) -> Option<BoxAdmission> {
    additive(oracle).required(offset)
}
fn box_correction<const D: usize>(oracle: &Oracle<D>, lower: f64) -> Option<BoxAdmission> {
    BoxAdmission::contribution(
        lower,
        oracle.network.multiplicity.lower,
        oracle.multiplicity_upper(),
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/box_pricing_tests.rs"]
mod box_pricing_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/early_witness_tests.rs"]
mod early_witness_tests;

/// Native cost sign precision is 1e-6. Unlike snapping those costs to zero,
/// retain a lower estimate covering every walk in the actual pricing domain.
fn near_zero_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    tolerance: f64,
) -> Option<(f64, f64)> {
    additive(oracle)
        .near_zero(offset, tolerance)
        .map(|bound| (bound.minimum, bound.correction))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/grouped_box_tests.rs"]
mod grouped_box_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/coupled_admission_tests.rs"]
mod coupled_admission_tests;

/// Prepare the shared oracle at the exact dual point used by fixing.
pub(super) fn cache_duals<const D: usize>(
    state: &mut State<D>,
    index: usize,
    duals: &[f64],
) -> Result<f64> {
    use crate::master::pricing::Networks;
    let tolerance = state.scheduler.tolerance;
    // Retained-pool pricing normally skips labeling preparation. Fixing and
    // inspection must instead refresh costs at these raw duals, even when a
    // preceding gap pass left a complete retained pool.
    let retained = state.oracles[index].solver.enumeration.take();
    let coverage = state.gap_pool_cutoff.take();
    let universe = state.universe.take();
    let projected = retirement::duals(state, duals);
    let prepared = NetworkPricingAdapter { state, tolerance }.prepare(index, 0, &projected, false);
    state.oracles[index].solver.enumeration = retained;
    state.gap_pool_cutoff = coverage;
    state.universe = universe;
    let request = prepared?;
    let oracle = &mut state.oracles[index];
    oracle
        .solver
        .cache_physical_costs(&oracle.costs, &oracle.costs)?;
    Ok(request.offset)
}
