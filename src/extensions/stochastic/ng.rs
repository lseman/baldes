//! Cheap NG-route relaxation: scalar travel pricing followed by true recourse replay.
use super::pricing::route_uses;
use super::{ArcBranch, Policy, PricedRoute, PricingOptions, PricingStatus, Problem};
use crate::{invalid, Result};
use std::collections::BTreeSet;
use std::time::Instant;

#[cfg(test)]
#[path = "../../../tests/unit/extensions/stochastic/ng_proof_sweep.rs"]
mod proof_sweep_tests;

#[derive(Clone, Debug)]
pub struct NgPricingResult {
    /// Complete means the final scalar NG relaxation was completely priced.
    /// It does not establish convergence of stochastic column generation.
    pub status: PricingStatus,
    /// Only elementary, mean-capacity-feasible columns with true recourse costs.
    pub routes: Vec<PricedRoute>,
    /// Global lower bound on stochastic reduced costs, never an observed minimum
    /// from an incomplete search. Negative nonmetric recourse is included.
    pub lower_bound: Option<f64>,
    pub statistics: crate::labeling::Statistics,
    pub refinements: usize,
    pub discarded_walks: usize,
}

impl Problem {
    /// Price a travel-plus-recourse-lower-bound NG relaxation. Cyclic walks
    /// provide relaxation bounds only; they never become master columns.
    /// When cycles obscure all improving elementary candidates, grow NG memory
    /// along those cycles (decremental state-space relaxation) and reprice.
    #[allow(clippy::too_many_arguments)]
    pub fn price_ng(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        neighborhood: usize,
        options: PricingOptions,
    ) -> Result<NgPricingResult> {
        self.price_ng_scheduled(
            policy,
            customer_duals,
            fleet_dual,
            branches,
            branch_duals,
            phase_one,
            neighborhood,
            options,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn price_ng_scheduled(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        neighborhood: usize,
        options: PricingOptions,
        complete_first: bool,
    ) -> Result<NgPricingResult> {
        if neighborhood == 0 {
            return Err(invalid("NG neighborhood must be positive"));
        }
        let started = Instant::now();
        let n = self.demands.len();
        let mut graph = self.pricing_graph(
            policy,
            customer_duals,
            fleet_dual,
            branches,
            branch_duals,
            phase_one,
        )?;
        graph.set_coupled_resources(&[])?;
        graph.elementary = false;
        if self.demands.iter().any(|d| d.mean() > 0.) {
            graph.main_resource = Some(1);
        }
        for v in 1..=n {
            let mut nearest: Vec<_> = (1..=n).filter(|&j| j != v).collect();
            nearest.sort_by(|&a, &b| {
                (self.costs[v][a] + self.costs[a][v])
                    .total_cmp(&(self.costs[v][b] + self.costs[b][v]))
                    .then(a.cmp(&b))
            });
            graph.vertices[v].memory = std::iter::once(v - 1)
                .chain(
                    nearest
                        .into_iter()
                        .take(neighborhood.saturating_sub(1))
                        .map(|j| j - 1),
                )
                .collect();
        }
        let charges = self.recourse_customer_lower_bounds();
        for arc in &mut graph.arcs {
            if !phase_one && arc.head != graph.sink {
                let charge = charges[arc.head - 1];
                arc.cost += charge;
                arc.reduced_cost += charge;
            }
        }
        let mut result = NgPricingResult {
            status: PricingStatus::WorkLimit,
            routes: vec![],
            lower_bound: None,
            statistics: Default::default(),
            refinements: 0,
            discarded_walks: 0,
        };
        if n == 0 {
            result.status = PricingStatus::Complete;
            return Ok(result);
        }
        let mut orders = BTreeSet::new();
        let mut labels_left = options.max_labels;
        // Mirror the default CVRP/VRPTW staged pricer: try a jump-arc NG
        // phase first, then progressively relax bucket truncation. These
        // phases only generate candidate columns; they never certify bounds.
        let mut heuristic_stage = (n > 9 && !complete_first).then_some(0usize);
        while labels_left > 0 && options.max_columns > 0 && n > 0 {
            let remaining = options.time_limit.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                result.status = PricingStatus::TimeLimit;
                break;
            }
            let mut solver = crate::pricing::Solver::<3>::new(graph.vertices.len(), 2)?;
            solver.graph = graph.clone();
            if let Some(stage) = heuristic_stage {
                solver.phases = vec![crate::pricing::phases::Phase {
                    check_ng: true,
                    jump_arcs: stage == 0,
                    labels_per_bucket: [16, 4, 1][stage],
                }];
            }
            solver.controls.max_labels = labels_left;
            solver.controls.max_columns = options.max_columns;
            solver.controls.max_route_arcs = Some(n + 1);
            solver.controls.time_limit = remaining;
            // One directional sweep keeps the configured single-thread budget.
            solver.set_option("BALDESuseBidirectionalSearch", 0.)?;
            let priced = match solver.price_phase_partial(0, options.max_columns, false) {
                Ok(p) => p,
                Err(e) if e.message.contains("time limit") => {
                    result.status = PricingStatus::TimeLimit;
                    break;
                }
                Err(e) if e.message.contains("limit") => break,
                Err(e) => return Err(e),
            };
            labels_left = labels_left.saturating_sub(priced.statistics.generated.max(1));
            result.statistics.absorb(&priced.statistics);
            if priced.exact && !priced.columns.is_empty() {
                let minimum = priced.minimum_reduced_cost;
                let bound = minimum - 1e-8 * (1. + minimum.abs());
                result.lower_bound = Some(result.lower_bound.map_or(bound, |b| b.max(bound)));
            }
            let mut tightened = false;
            for column in priced.columns {
                let route: Vec<_> = column
                    .nodes
                    .iter()
                    .skip(1)
                    .take(column.nodes.len() - 2)
                    .map(|&j| j as usize)
                    .collect();
                let mut previous = vec![None; n + 1];
                let mut elementary = true;
                for (position, &j) in route.iter().enumerate() {
                    if let Some(before) = previous[j] {
                        elementary = false;
                        // Remember j on the entire repeated segment. This
                        // removes this walk while retaining all elementary paths.
                        for &v in &route[before..=position] {
                            if !graph.vertices[v].memory.contains(&(j - 1)) {
                                graph.vertices[v].memory.push(j - 1);
                                tightened = true;
                            }
                        }
                    }
                    previous[j] = Some(position);
                }
                if !elementary {
                    result.discarded_walks += 1;
                    continue;
                }
                let load: f64 = (1..=n)
                    .filter(|j| route.contains(j))
                    .map(|j| self.demands[j - 1].mean())
                    .sum();
                if load > self.capacity as f64 {
                    continue;
                }
                let evaluation = self.evaluate_route(&route, policy)?;
                let mut reduced_cost =
                    if phase_one { 0. } else { evaluation.total_cost } - fleet_dual;
                for &j in &route {
                    reduced_cost -= customer_duals[j - 1];
                }
                for (&b, &dual) in branches.iter().zip(branch_duals) {
                    if route_uses(&route, b) {
                        reduced_cost -= dual;
                    }
                }
                if !reduced_cost.is_finite() {
                    return Err(invalid("NG recourse replay overflow"));
                }
                if reduced_cost < -1e-7 && orders.insert(route.clone()) {
                    result.routes.push(PricedRoute {
                        customers: route,
                        evaluation,
                        reduced_cost,
                    });
                }
            }
            result.status = if started.elapsed() >= options.time_limit {
                PricingStatus::TimeLimit
            } else if priced.exact {
                PricingStatus::Complete
            } else {
                PricingStatus::WorkLimit
            };
            if !result.routes.is_empty() {
                if heuristic_stage.is_some() && result.status != PricingStatus::TimeLimit {
                    result.status = PricingStatus::Heuristic;
                }
                break;
            }
            if let Some(stage) = heuristic_stage {
                // Continue through the default-style schedule while work
                // remains. Once bounded phases are exhausted, run a complete
                // scalar NG relaxation before exact stochastic pricing.
                if result.status == PricingStatus::TimeLimit || labels_left == 0 {
                    break;
                }
                heuristic_stage = (stage < 2).then_some(stage + 1);
                if tightened {
                    result.refinements += 1;
                }
                continue;
            }
            if !priced.exact || !tightened {
                break;
            }
            result.refinements += 1;
        }
        result
            .routes
            .sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
        result.statistics.elapsed = started.elapsed().as_secs_f64();
        Ok(result)
    }
}
