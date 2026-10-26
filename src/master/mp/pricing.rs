//! Per-network pricing and shared-master controller callbacks.
use super::*;

impl<const D: usize> cg::Controller<ClpMaster> for Coordinator<D> {
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
        true
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
        if self.universe.is_some() {
            self.scheduler.complete = true;
            return enumeration::price(self, duals);
        }
        let mut columns = vec![];
        let mut minimum = f64::INFINITY;
        let mut exact = true;
        let mut correction = 0.;
        let mut gradient = self.rhs.clone();
        let active = self
            .oracles
            .iter()
            .enumerate()
            .filter_map(|(index, oracle)| (oracle.multiplicity_upper() != 0.).then_some(index))
            .collect();
        let order = self.scheduler.order(phase, active);
        let mut visited = 0;
        for &index in &order {
            let oracle = &mut self.oracles[index];
            let remaining = self.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(incomplete("multiple-network pricing time limit"));
            }
            oracle.solver.controls.time_limit = remaining;
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
                            cost - duals[self.shared_offset + row] * value
                        })
                    }),
            );
            for &(cut, row) in &oracle.nonlinear_rows {
                oracle.solver.cuts.cuts[cut].dual = duals[self.shared_offset + row];
            }
            for &(cut, row) in &oracle.knapsack_rows {
                oracle.solver.cuts.knapsack_cuts[cut].dual = duals[self.shared_offset + row];
            }
            if !oracle.presence_rows.is_empty() {
                let mut inputs = oracle.solver.presence_rows().to_vec();
                for &(input, row) in &oracle.presence_rows {
                    inputs[input].dual = duals[self.shared_offset + row];
                }
                oracle.solver.set_presence_rows(inputs)?;
            }
            oracle.solver.cache_variable_costs(&oracle.costs)?;
            let offset = oracle
                .route_entries
                .iter()
                .map(|&(row, value)| -duals[row] * value)
                .sum::<f64>()
                - oracle
                    .network
                    .route_entries
                    .iter()
                    .map(|&(row, value)| duals[self.shared_offset + row] * value)
                    .sum::<f64>();
            oracle.calls += 1;
            let result = oracle
                .solver
                .price_functor_phase(phase, oracle.solver.controls.max_columns)?;
            if oracle.solver.phases[phase].exact() && !result.exact {
                return Err(incomplete("network exact pricing did not finish"));
            }
            exact &= result.exact;
            let reduced = result.minimum_reduced_cost + offset;
            if !reduced.is_finite() {
                return Err(incomplete("network minimum is not finite"));
            }
            if result.columns.is_empty() && (reduced < 0. || oracle.network.multiplicity.lower > 0.)
            {
                return Err(incomplete("required network pricing witness unavailable"));
            }
            minimum = minimum.min(reduced);
            self.scheduler.record(index, reduced);
            visited += 1;
            let mut new_improving_columns = 0;
            let quantity = if reduced >= 0. {
                oracle.network.multiplicity.lower
            } else {
                oracle.multiplicity_upper()
            };
            correction += quantity * reduced;
            let mut best: Option<(f64, Vec<(usize, f64)>)> = None;
            for mut route in result.columns {
                let coefficients = entries(oracle, &route, self.shared_offset)?;
                let mut ids = route.arcs.clone();
                ids.sort_unstable();
                let cost = ids
                    .iter()
                    .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                    .sum::<f64>()
                    + oracle
                        .network
                        .graph
                        .coupled_path_cost(
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
                let tag = if let Some(&tag) = self.tags.get(&key) {
                    tag
                } else {
                    new_improving_columns += usize::from(reduced_cost < -self.scheduler.tolerance);
                    let tag = i32::try_from(self.routes.len())
                        .map_err(|_| invalid("network route tag overflow"))?;
                    self.tags.insert(key, tag);
                    self.routes.push(Arc::new(NetworkRoute {
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
            if let Some((_, coefficients)) = best {
                for (row, value) in coefficients {
                    gradient[row] -= quantity * value;
                }
            }
            if self.scheduler.finish_network(index, new_improving_columns) {
                break;
            }
        }
        self.scheduler.complete = visited == order.len();
        exact &= self.scheduler.complete;
        if minimum == f64::INFINITY {
            minimum = 0.;
        }
        if !self.scheduler.complete {
            // Unvisited minima are unknown at these duals. Cached scores are
            // ordering hints only; do not publish a partial correction/gradient.
            return Ok(cg::Answer {
                columns,
                minimum,
                exact: false,
                bound_minimum: None,
            });
        }
        // Complete heuristic batches also supply the estimate for smoothing.
        // The CG driver publishes it as a bound only for an exact batch.
        if let Some((static_correction, static_gradient)) =
            master_variables::correction(self, duals)
        {
            correction += static_correction;
            for (row, value) in static_gradient {
                gradient[row] -= value;
            }
            if correction.is_finite() && gradient.iter().all(|g| g.is_finite()) {
                self.correction = Some(correction);
                self.gradient = Some(gradient);
            } else {
                self.correction = None;
                self.gradient = None;
            }
        } else {
            // A negative reduced cost on an unbounded static component leaves
            // this dual point's estimate unknown. Still return priced routes.
            self.correction = None;
            self.gradient = None;
        }
        Ok(cg::Answer {
            columns,
            minimum,
            exact,
            bound_minimum: None,
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/native_contribution_tests.rs"]
mod native_contribution_tests;
