//! Per-network pricing and shared-master controller callbacks.
use super::*;
use crate::master::LpMaster;

impl<const D: usize> cg::Controller<ClpMaster> for Coordinator<D> {
    fn master_solved(
        &mut self,
        master: &ClpMaster,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        self.3 = artificials
            .iter()
            .try_fold(f64::INFINITY, |minimum, &index| {
                let cost = *master.costs().get(index)?;
                (cost.is_finite() && cost > 0.).then_some(minimum.min(cost))
            });
        let mut state = self.0.borrow_mut();
        let objective = master.objective();
        state.pricing_master =
            (master.solved() && objective.is_finite() && master.objective_offset.is_finite())
                .then_some((objective, master.objective_offset));
        if !state.integer_primal_termination {
            return Ok(());
        }
        if let Some(witness) = tree_adapter::replay_integer_witness(
            &state,
            master,
            |column| tags.get(column).copied(),
            state.witness_tolerance,
        )? {
            if state
                .pricing_witness
                .as_ref()
                .map_or(true, |old| witness.0 < old.0)
            {
                state.pricing_witness = Some(witness);
            }
        }
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        self.0
            .borrow()
            .pricing_witness
            .as_ref()
            .map_or(f64::INFINITY, |w| w.0)
    }
    fn original_witness_available(&self) -> bool {
        self.0.borrow().pricing_witness.is_some()
    }
    fn curvature_rows(&self, row_count: usize) -> Vec<cg::CurvatureRow> {
        let state = self.0.borrow();
        (0..row_count)
            .map(|row| {
                state
                    .curvature_rows
                    .get(row)
                    .copied()
                    .unwrap_or(cg::CurvatureRow {
                        class: cg::CurvatureRowClass::Excluded,
                        active: false,
                    })
            })
            .collect()
    }
    fn recover_feasibility(
        &mut self,
        master: &ClpMaster,
        warm: &cg::WarmStart,
        remaining: std::time::Duration,
        iterations: usize,
        tolerance: f64,
        bound_mode: cg::BoundMode,
    ) -> Result<Option<cg::FeasibilityRecovery>> {
        super::feasibility::recover(
            &self.0, master, warm, remaining, iterations, tolerance, bound_mode,
        )
        .map(Some)
    }
    fn pricing_complete(&self) -> bool {
        self.0.borrow().scheduler.complete
    }
    fn request_enumeration_checkpoint(&mut self, _master: &ClpMaster) -> Result<bool> {
        let requested = self.1 && self.0.borrow().universe.is_none();
        // A failed attempt must resume ordinary pricing, not request another
        // checkpoint at the same point. A later solve may retry with new limits.
        self.1 = false;
        Ok(requested)
    }
    fn column_upper_bound(&self, tag: i32) -> Option<f64> {
        let state = self.0.borrow();
        if tag >= 0 {
            // Ordinary generated columns share the network multiplicity row.
            // Its auxiliary violation must cost at least as much as every
            // uncapped column can improve the objective; otherwise preserve
            // the cap until penalties have increased enough.
            // Repeating that limit on every column is redundant and can move
            // the convexity dual onto bound-active columns, leaving pricing
            // with an artificially weak network offset. Explicit restrictions
            // and finite retained-universe boxes still need their own caps.
            let covered = self.3.is_some_and(|penalty| {
                penalty == f64::INFINITY
                    || state.routes.get(tag as usize).is_some_and(|route| {
                        crate::core::rounding::add_down(route.route.original_cost, penalty)
                            .is_some_and(|sum| sum >= 0.)
                    })
            });
            if covered
                && state.master_variables.is_empty()
                && state.shared_rows.is_empty()
                && state.variable_boxes.is_empty()
                && state.grouped_variable_boxes.is_empty()
                && state.synchronized_services.is_empty()
                && state.universe.is_none()
                && !state.route_upper_bounds.contains_key(&(tag as usize))
            {
                return None;
            }
            enumeration::route_upper(&state, tag as usize).filter(|u| u.is_finite())
        } else if let Some((index, sign)) = master_variables::decode(tag) {
            state
                .master_variables
                .get(index)
                .map(|v| master_variables::upper(v, sign))
                .filter(|u| u.is_finite())
        } else {
            None
        }
    }
    fn certified_pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        Ok(Some(
            bounds::certificate(&self.0.borrow(), master, duals).unwrap_or(f64::NEG_INFINITY),
        ))
    }
    fn pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        let state = self.0.borrow();
        Ok(Some(match state.correction {
            Some(correction) => {
                duals
                    .iter()
                    .zip(&state.rhs)
                    .fold(master.objective_offset, |value, (dual, rhs)| {
                        value + dual * rhs
                    })
                    + correction
            }
            None => f64::NEG_INFINITY,
        }))
    }
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.0.borrow().gradient.clone()
    }
    fn take_pricing_subgradient(&mut self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.0.borrow_mut().gradient.take()
    }
    fn uses_inactive_column_pool(&self) -> bool {
        self.2
    }
    fn reprice_inactive_column(
        &mut self,
        column: &cg::Column,
        duals: &[f64],
    ) -> Result<Option<cg::Column>> {
        let state = self.0.borrow();
        let Some(tag) = usize::try_from(column.tag).ok() else {
            return Ok(None);
        };
        let Some(route) = state.routes.get(tag) else {
            return Ok(None);
        };
        let Some(oracle) = state.oracles.iter().find(|o| o.network.id == route.network) else {
            return Ok(None);
        };
        if oracle.multiplicity_upper() == 0.
            || state
                .universe
                .as_ref()
                .is_some_and(|universe| !universe.contains(&tag))
            || route.route.arcs.iter().any(|id| {
                oracle
                    .proved_zero_arcs
                    .get(oracle.index[id])
                    .copied()
                    .unwrap_or(false)
            })
            || !lifecycle::integer_route_feasible(oracle, &route.route).unwrap_or(false)
        {
            return Ok(None);
        }
        let entries = entries(oracle, &route.route, state.shared_offset)?;
        let cost = route.route.original_cost;
        let reduced_cost = entries
            .iter()
            .fold(cost, |value, &(row, a)| value - duals[row] * a);
        Ok(Some(cg::Column {
            cost,
            reduced_cost,
            entries,
            acceptable: true,
            tag: column.tag,
            identity: column.identity.clone(),
        }))
    }
}

impl<const D: usize> State<D> {
    pub(super) fn price(&mut self, phase: usize, duals: &[f64]) -> Result<cg::Answer> {
        self.correction = None;
        self.gradient = None;
        self.scheduler.complete = false;
        let rhs = self.rhs.clone();
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
            &mut NetworkAdapter {
                state: self,
                tolerance,
            },
            &mut scheduler,
            context,
        );
        self.scheduler = scheduler;
        let batch = result?;
        self.correction = batch.correction;
        self.gradient = batch.gradient;
        Ok(batch.answer)
    }
}

struct NetworkAdapter<'a, const D: usize> {
    state: &'a mut State<D>,
    tolerance: f64,
}
impl<const D: usize> crate::master::pricing::Networks<D> for NetworkAdapter<'_, D> {
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
        if self.state.universe.is_some() || control_pricing::needed(oracle) {
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
            let mut order: Vec<usize> = (0..registry.cuts.len())
                .filter(|&index| {
                    !registry.retired.contains(&index)
                        && !registry.cuts[index].visit_once
                        && registry.cuts[index].dual.abs() > 1e-6
                })
                .collect();
            crate::cuts::shared::stdsort::stable_sort(&mut order, |&a, &b| {
                let (va, vb) = (-registry.cuts[a].dual, -registry.cuts[b].dual);
                va > vb + 1e-6 || (!(vb - 1e-6 > va) && a < b)
            });
            registry.pricing_order = order;
            registry.round_duals = true;
        }
        for &(cut, row) in &oracle.knapsack_rows {
            oracle.solver.cuts.knapsack_cuts[cut].dual = duals[self.state.shared_offset + row];
        }
        if !oracle.presence_rows.is_empty() {
            let mut inputs = oracle.solver.presence_rows().to_vec();
            for &(input, row) in &oracle.presence_rows {
                inputs[input].dual = duals[self.state.shared_offset + row];
            }
            oracle.solver.set_presence_rows(inputs)?;
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
        if self.state.universe.is_none() && control_pricing::needed(&self.state.oracles[index]) {
            return Some((|| {
                let (routes, minimum) = control_pricing::price(self.state, index, duals)?;
                self.state.oracles[index].calls += 1;
                let collected = self.collect(index, routes, 0., duals)?;
                let upper = self.state.oracles[index].multiplicity_upper();
                let columns = collected
                    .columns
                    .into_iter()
                    .map(|column| {
                        let cap = self
                            .state
                            .route_upper_bounds
                            .get(&(column.tag as usize))
                            .copied()
                            .unwrap_or(upper)
                            .min(upper);
                        (column, cap)
                    })
                    .collect();
                Ok(crate::master::pricing::Retained {
                    minimum_bound: Some(minimum),
                    columns,
                    independent_caps: false,
                    maximum: usize::MAX,
                })
            })());
        }
        let universe = self.state.universe.as_ref()?;
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
                let coefficients = entries(oracle, &route.route, self.state.shared_offset)?;
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
        let mut columns = vec![];
        let mut new_improving_columns = 0;
        let mut best: Option<(f64, Vec<(usize, f64)>)> = None;
        for mut route in routes {
            let coefficients = entries(oracle, &route, self.state.shared_offset)?;
            let mut ids = route.arcs.clone();
            ids.sort_unstable();
            let cost = ids
                .iter()
                .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                .sum::<f64>()
                + route
                    .coupled_cost(
                        &oracle.network.graph,
                        &route
                            .arcs
                            .iter()
                            .map(|id| oracle.index[id])
                            .collect::<Vec<_>>(),
                    )?
                    .ok_or_else(|| invalid("network path violates coupled resources"))?;
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
                .map_or(true, |(value, _)| reduced_cost < *value)
            {
                best = Some((reduced_cost, coefficients.clone()));
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
                self.state.tags.insert(key, tag);
                self.state.routes.push(Arc::new(NetworkRoute {
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

        Ok(crate::master::pricing::Collected {
            columns,
            fresh: new_improving_columns,
            best_entries: best.map(|(_, entries)| entries),
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
