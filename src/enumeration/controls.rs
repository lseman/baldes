//! Enumeration limits and arc-path preservation policy.
use std::time::Duration;
#[derive(Clone, Debug)]
pub struct Controls {
    pub label_limit: usize,
    pub path_limit: usize,
    pub partitions: usize,
    pub cutoff: f64,
    pub time_limit: Duration,
    pub max_route_arcs: Option<usize>,
    /// Retain every arc ordering for current and future arc-flow branch rows.
    pub preserve_arc_paths: bool,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            label_limit: 1_000_000,
            path_limit: 1_000_000,
            partitions: 1024,
            cutoff: 0.,
            time_limit: Duration::from_secs(1000),
            max_route_arcs: None,
            preserve_arc_paths: false,
        }
    }
}

impl Controls {
    pub(crate) fn for_graph<const D: usize>(&self, graph: &crate::Graph<D>) -> crate::Result<Self> {
        let mut controls = self.clone();
        if let Some(limit) = graph.max_route_arcs() {
            if controls.max_route_arcs.is_some_and(|old| old < limit) {
                return Err(crate::invalid(
                    "an enumeration arc cap cannot narrow the declared bounded-walk domain",
                ));
            }
            controls.max_route_arcs =
                Some(controls.max_route_arcs.map_or(limit, |old| old.min(limit)));
            controls.preserve_arc_paths = true;
        }
        Ok(controls)
    }
}
