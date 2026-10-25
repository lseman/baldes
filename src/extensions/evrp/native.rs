//! EV model adapter for the ordinary physical-graph pricing engine.
use super::*;
use crate::graph::coupled::{ChargingResourceArc, CoupledResource, PiecewiseLinearFunction};
use crate::{Graph, Window};

#[derive(Clone, Copy, Debug)]
pub struct NativeOptions {
    pub max_arcs: usize,
    pub max_labels: usize,
    /// Budget per charging-front extension (also used during suffix joins).
    pub max_front_work: usize,
    /// Budget for reconstructing the selected stop order's charging witness.
    pub max_reconstruction_work: usize,
    pub bidirectional: bool,
}
impl Default for NativeOptions {
    fn default() -> Self {
        Self {
            max_arcs: 16,
            max_labels: 20_000,
            max_front_work: 2_000_000,
            max_reconstruction_work: 1_000_000,
            bidirectional: true,
        }
    }
}
impl NativeOptions {
    pub(super) fn validation_options(self) -> DirectOptions {
        DirectOptions {
            max_arcs: self.max_arcs,
            max_labels: self.max_labels,
            max_work: self.max_front_work,
            dominance: true,
        }
    }
}

impl Problem {
    /// Compile to a native physical graph. Original location IDs are preserved;
    /// one additional vertex duplicates the depot as the sink. Only customers
    /// receive elementarity sets. Stations remain repeatable. Native exact NG
    /// memory currently permits at most fifteen customers in this adapter.
    /// The scalar coordinates are arc count and delivered load. A native PWL
    /// state map records whether a customer was served, excluding empty routes.
    /// Callers can attach native cuts/restrictions through `pricing::Solver`.
    pub fn native_graph(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        departure: f64,
        options: NativeOptions,
    ) -> Result<Graph<3>> {
        self.validate_pricing(duals, fleet_dual, departure, options.validation_options())?;
        let n = self.locations.len();
        let customers = self
            .locations
            .iter()
            .filter(|l| matches!(l.kind, LocationKind::Customer { .. }))
            .count();
        if customers > 15 || options.max_arcs as u128 > (1u128 << 52) {
            return Err(invalid("native EV pricing supports at most fifteen customers and an exactly representable arc bound"));
        }
        let mut graph = Graph::<3>::new(
            n.checked_add(1)
                .ok_or_else(|| invalid("EV graph size overflow"))?,
            2,
        )?;
        graph.source = self.depot;
        graph.sink = n;
        graph.elementary = true;
        graph.main_resource = Some(0);
        for vertex in &mut graph.vertices {
            vertex.elements.clear();
            vertex.memory.clear();
        }
        graph.element_count = 0;
        for (i, location) in self.locations.iter().enumerate() {
            if matches!(location.kind, LocationKind::Customer { .. }) {
                graph.add_elementarity_set(&[i])?;
            }
        }
        for v in 0..=n {
            graph.set_bounds(v, 0, 0., options.max_arcs as f64)?;
            graph.set_bounds(
                v,
                1,
                0.,
                self.load_capacity + crate::graph::coupled::DEFAULT_LOAD_TOLERANCE,
            )?;
        }
        for r in 0..2 {
            graph.initial[r] = Some(0.);
            graph.set_bounds(self.depot, r, 0., 0.)?;
        }
        let mut arcs = Vec::new();
        let mut served = Vec::new();
        for (from, row) in self.travel.iter().enumerate() {
            if from != self.depot && matches!(self.locations[from].kind, LocationKind::Depot) {
                continue;
            }
            for (to, travel) in row.iter().enumerate() {
                let Some(travel) = travel else {
                    continue;
                };
                if from == to
                    || (to != self.depot && matches!(self.locations[to].kind, LocationKind::Depot))
                {
                    continue;
                }
                let head = if to == self.depot { n } else { to };
                let id = graph.add_arc(from, head, travel.cost)?;
                let arc = graph.arc_mut(id)?;
                arc.reduced_cost =
                    travel.cost - duals[to] - if from == self.depot { fleet_dual } else { 0. };
                arc.consumption[0] = 1.;
                if let LocationKind::Customer { demand } = self.locations[to].kind {
                    arc.consumption[1] = demand;
                }
                // Boolean state: a customer sets it to one; other stops
                // preserve it. The sink accepts only an already-set state.
                let points = if head == n {
                    vec![[1., 1.], [2., 1.]]
                } else if matches!(self.locations[to].kind, LocationKind::Customer { .. }) {
                    vec![[0., 1.], [1., 1.]]
                } else {
                    vec![[0., 0.], [1., 1.]]
                };
                served.push(PiecewiseLinearFunction::new(points)?);
                let location = &self.locations[to];
                let charging = match &location.kind {
                    LocationKind::Station { curve } => Some(PiecewiseLinearFunction::new(
                        curve.knots.iter().map(|&(e, t)| [e, t]).collect(),
                    )?),
                    _ => None,
                };
                arcs.push(ChargingResourceArc {
                    travel_time: travel.time,
                    energy: travel.energy,
                    window: Window::new(location.window.0, location.window.1)?,
                    service: location.service,
                    charging,
                });
            }
        }
        // The ordinary engine reserves NO_ROUTE_COST as its pool/arc
        // sentinel. Reject unsupported objective scales instead of silently
        // omitting a positive route or a costly arc with a negative suffix.
        let cost_bound = graph
            .arcs
            .iter()
            .map(|a| a.reduced_cost.abs())
            .fold(0., f64::max)
            * options.max_arcs as f64;
        if !cost_bound.is_finite() || cost_bound >= crate::labeling::NO_ROUTE_COST - 1. {
            return Err(invalid("native EV reduced-cost magnitude times max_arcs must be below 1e12; rescale cost units"));
        }
        let depot = &self.locations[self.depot];
        // An unavailable initial service start is represented by an empty
        // source adjacency, rather than an invented feasible initial state.
        if departure.max(depot.window.0) > depot.window.1 {
            graph.arcs.clear();
            arcs.clear();
            served.clear();
        }
        let initial_time = departure.max(depot.window.0) + depot.service;
        graph.set_coupled_resources(&[
            // Scalar load supplies bucket bounds. The native load state also
            // guards feasibility/dominance without scalar EPSILON comparisons.
            CoupledResource::SimultaneousPickupDelivery {
                capacity: self.load_capacity,
                deliveries: graph.arcs.iter().map(|arc| arc.consumption[1]).collect(),
                pickups: vec![0.; graph.arcs.len()],
                tolerance: crate::graph::coupled::DEFAULT_LOAD_TOLERANCE,
            },
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window::new(0., 1.)?,
                functions: served,
            },
            CoupledResource::Charging {
                battery: self.battery,
                initial_energy: self.battery,
                initial_time,
                arcs,
                max_work: options.max_front_work,
            },
        ])?;
        Ok(graph)
    }

    /// Price through the native labeling engine, with continuous charging and
    /// customer elementarity. Coverage has the same bounded/global meaning as
    /// the reference backends. The chosen order is independently scheduled and
    /// strictly replayed before it becomes an EV column.
    pub fn price_native(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        departure: f64,
        options: NativeOptions,
    ) -> Result<EvPricingResult> {
        let mut result = EvPricingResult {
            status: EvPricingStatus::WorkLimit,
            best_route: None,
            statistics: Default::default(),
        };
        self.validate_pricing(duals, fleet_dual, departure, options.validation_options())?;
        let depot = &self.locations[self.depot];
        let start = departure.max(depot.window.0);
        if start > depot.window.1 {
            result.status = EvPricingStatus::Complete;
            return Ok(result);
        }
        if !(start + depot.service).is_finite() {
            result.status = EvPricingStatus::NumericalFailure;
            return Ok(result);
        }
        let graph = self.native_graph(duals, fleet_dual, departure, options)?;
        if options.max_labels == 0 || options.max_front_work == 0 {
            return Ok(result);
        }
        let mut solver = crate::pricing::Solver::<3>::new(graph.vertices.len(), 2)?;
        solver.graph = graph;
        solver.controls.max_labels = options.max_labels;
        solver.controls.max_route_arcs = Some(options.max_arcs);
        solver.set_option(
            "BALDESuseBidirectionalSearch",
            if options.bidirectional { 1. } else { 0. },
        )?;
        let priced = match solver.price_phase(0, 1, false) {
            Ok(priced) => priced,
            Err(error) => {
                result.status = if error.message.contains("limit") {
                    EvPricingStatus::WorkLimit
                } else {
                    EvPricingStatus::NumericalFailure
                };
                return Ok(result);
            }
        };
        result.statistics.labels = priced.statistics.generated;
        result.statistics.dominance_checks = priced.statistics.dominance_checks;
        result.statistics.joins = priced.statistics.concatenation_tests;
        if !priced.exact {
            return Ok(result);
        }
        if let Some(route) = priced.columns.first() {
            let order: Vec<_> = route
                .nodes
                .iter()
                .map(|&v| {
                    if v as usize == self.locations.len() {
                        self.depot
                    } else {
                        v as usize
                    }
                })
                .collect();
            let reconstructed = self.optimize_fixed_order(
                &order,
                departure,
                ChargingOptions {
                    max_work: options.max_reconstruction_work,
                },
            )?;
            result.statistics.work = reconstructed.work;
            match (reconstructed.status, reconstructed.solution) {
                (ChargingStatus::Optimal, Some(charging)) => {
                    let reduced_cost = charging
                        .visits
                        .iter()
                        .fold(charging.schedule.travel_cost - fleet_dual, |cost, v| {
                            cost - duals[v.location]
                        });
                    if !reduced_cost.is_finite() {
                        result.status = EvPricingStatus::NumericalFailure;
                        return Ok(result);
                    }
                    result.best_route = Some(PricedRoute {
                        charging,
                        reduced_cost,
                    });
                }
                (ChargingStatus::WorkLimit, _) => return Ok(result),
                _ => {
                    result.status = EvPricingStatus::NumericalFailure;
                    return Ok(result);
                }
            }
        }
        let depot = &self.locations[self.depot];
        result.status = self.pricing_coverage(
            departure.max(depot.window.0) + depot.service,
            options.max_arcs,
        );
        Ok(result)
    }
}
