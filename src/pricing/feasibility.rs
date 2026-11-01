//! Optional full-route feasibility checks at enumeration export.
use super::*;

/// Application feasibility predicate for reconstructed elementary routes.
///
/// This is an enumeration-domain restriction, evaluated before retained paths
/// are published. It does not filter ordinary labeling outputs or alter their
/// minimum reduced cost. Implementations must give stable answers for a route
/// until the check is replaced with `set_enumeration_feasibility_check`.
pub trait EnumerationFeasibilityCheck: std::fmt::Debug + Send + Sync {
    fn is_feasible(&self, route: &crate::Route) -> Result<bool>;
}

impl<const D: usize> Solver<D> {
    /// Replace the optional enumeration predicate and discard retained paths,
    /// labels and reduced-cost fixing computed for the previous domain.
    /// Node forks share the predicate and independently retain accepted paths.
    pub fn set_enumeration_feasibility_check(
        &mut self,
        check: Option<Arc<dyn EnumerationFeasibilityCheck>>,
    ) {
        self.restore_arcs();
        self.enumeration_feasibility = check;
    }
}
