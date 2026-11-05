//! Shared nominal-instance setup and physical route conversion for heuristics.
use super::{Instance, Rounding};
use crate::{Graph, Route};
use std::collections::HashMap;

impl Instance {
    /// Match the metric and time semantics of this instance's routing graph.
    /// Capacity-only graphs have zero time consumption, even for Solomon input.
    pub fn heuristic_instance<const D: usize>(&self, graph: &Graph<D>, rounding: Rounding) -> Self {
        let mut instance = self.clone();
        instance.rounding = rounding;
        if graph.resources.is_empty() || graph.arcs.iter().all(|a| a.consumption[0] == 0.) {
            for customer in &mut instance.customers {
                customer.ready = 0.;
                customer.due = f64::INFINITY;
                customer.service = 0.;
            }
        }
        instance
    }
}

pub(crate) struct HeuristicRouteMap {
    arcs: HashMap<(usize, usize), i32>,
}

impl HeuristicRouteMap {
    pub(crate) fn new<const D: usize>(graph: &Graph<D>) -> Self {
        Self {
            arcs: graph
                .arcs
                .iter()
                .map(|a| ((a.tail, a.head), a.id))
                .collect(),
        }
    }

    /// Replay instance customer IDs in the target split-depot graph. Keep the
    /// existing HGS mapping policy for parallel arcs (last physical arc wins).
    pub(crate) fn convert<const D: usize>(
        &self,
        graph: &Graph<D>,
        customers: &[usize],
    ) -> Option<Route> {
        let nodes: Vec<_> = std::iter::once(graph.source)
            .chain(customers.iter().copied())
            .chain(std::iter::once(graph.sink))
            .collect();
        let ids = nodes
            .windows(2)
            .map(|w| self.arcs.get(&(w[0], w[1])).copied())
            .collect::<Option<Vec<_>>>()?;
        graph.elementary_route(&ids).ok().flatten()
    }
}
