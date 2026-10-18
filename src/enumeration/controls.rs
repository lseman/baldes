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
