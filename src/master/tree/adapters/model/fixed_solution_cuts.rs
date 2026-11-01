//! Application-defined presence cuts separated from a dive's fixed solution.
use super::*;

/// Generated route columns only. A fixed identity is absent from the residual
/// list even when its previous LP quantity exceeds the quantity already fixed.
#[derive(Clone, Debug)]
pub struct FixedSolutionInput {
    pub residual_routes: Vec<FixedRoute>,
    pub fixed_routes: Vec<FixedRoute>,
}

/// One presence row valid for the application's original model. Coordinate and
/// finite-value checks are enforced by the ordinary row installer; mathematical
/// validity of a user-defined cut is the separator author's responsibility.
#[derive(Clone, Debug)]
pub struct FixedSolutionCut {
    /// Higher priorities outside the absolute/relative comparison band install
    /// first. Priorities tied within that band retain emission order.
    pub priority: f64,
    pub row: Row,
    pub coefficients: Vec<PresenceCoefficients>,
    pub master_coefficients: Vec<(i32, f64)>,
}

type Callback = dyn Fn(&FixedSolutionInput) -> Result<Vec<FixedSolutionCut>> + Send + Sync;

/// Runtime separator registration, shared safely across independent tree workers.
/// It runs before dive child CG, never during the initial greedy heuristic.
#[derive(Clone)]
pub struct FixedSolutionSeparator {
    pub name: String,
    callback: Arc<Callback>,
}
impl std::fmt::Debug for FixedSolutionSeparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedSolutionSeparator")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
impl FixedSolutionSeparator {
    pub fn new(
        name: impl Into<String>,
        callback: impl Fn(&FixedSolutionInput) -> Result<Vec<FixedSolutionCut>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            callback: Arc::new(callback),
        }
    }
}

fn input<const D: usize>(node: &NodeSnapshot<D>) -> FixedSolutionInput {
    let mut fixed_routes: Vec<FixedRoute> = Vec::new();
    let mut identities: HashMap<(i32, String), usize> = HashMap::new();
    for fixed in &node.state.fixed_routes {
        let key = (fixed.route.network, fixed.route.route.identity());
        if let Some(&index) = identities.get(&key) {
            fixed_routes[index].quantity += fixed.quantity;
        } else {
            identities.insert(key, fixed_routes.len());
            fixed_routes.push(fixed.clone());
        }
    }
    // RHS/bound changes intentionally invalidate the solved flag without
    // discarding the parent primal map. Native separation observes that map
    // before the new child's first column-generation evaluation.
    let values = node.master.values();
    let residual_routes = node
        .warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(column, metadata)| {
            let tag = usize::try_from(metadata.as_ref()?.tag).ok()?;
            let route = node.state.routes.get(tag)?;
            let quantity = *values.get(column)?;
            if quantity == 0. || identities.contains_key(&(route.network, route.route.identity())) {
                return None;
            }
            Some(FixedRoute {
                route: route.as_ref().clone(),
                quantity,
            })
        })
        .collect();
    FixedSolutionInput {
        residual_routes,
        fixed_routes,
    }
}

/// Returned rows live on a private node; no-op separation needs no copy.
/// A static-only fixed solution triggers the
/// lifecycle too, while the callback's route lists deliberately omit statics.
pub(super) fn prepare<const D: usize>(
    node: &NodeSnapshot<D>,
    options: &ModelTreeOptions,
    fixed_static: bool,
) -> Result<Option<NodeSnapshot<D>>> {
    if options.fixed_solution_separators.is_empty()
        || (node.state.fixed_routes.is_empty() && !fixed_static)
    {
        return Ok(None);
    }
    let input = input(node);
    let mut cuts = Vec::new();
    for separator in &options.fixed_solution_separators {
        cuts.extend((separator.callback)(&input)?);
    }
    if cuts.iter().any(|cut| !cut.priority.is_finite()) {
        return Err(invalid("fixed-solution cut priorities must be finite"));
    }
    if cuts.is_empty() {
        return Ok(None);
    }
    // Native primary comparison uses subtraction with the Double margin.
    // Approximate ties are not a transitive ordering, so do not hand this
    // comparator to sort_by. Stable insertion also retains emission order
    // where native instance-ID/priority-rule metadata is unavailable.
    let mut ordered: Vec<FixedSolutionCut> = Vec::with_capacity(cuts.len());
    for cut in cuts {
        let position = ordered
            .iter()
            .position(|old| {
                let margin = old.priority.abs().max(cut.priority.abs()) * 1e-10 + 1e-6;
                old.priority < cut.priority - margin
            })
            .unwrap_or(ordered.len());
        ordered.insert(position, cut);
    }
    let mut child = node.fork()?;
    for cut in ordered {
        child =
            child.with_presence_master_row(cut.row, &cut.coefficients, &cut.master_coefficients)?;
    }
    Ok(Some(child))
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/fixed_solution_cut_tests.rs"]
mod tests;
