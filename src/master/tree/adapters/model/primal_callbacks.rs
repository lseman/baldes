//! User primal heuristics consume original solution coordinates and return
//! candidates. Their claims never supply a lower bound or bypass replay.
use super::*;

/// Full original solution, including any fixed route contributions.
#[derive(Clone, Debug, Default)]
pub struct PrimalHeuristicSolution {
    pub routes: Vec<FixedRoute>,
    /// Omitted original master variables have value zero.
    pub master_values: Vec<(i32, f64)>,
}

/// Original-coordinate point supplied to a user primal heuristic.
#[derive(Debug)]
pub struct PrimalHeuristicInput {
    pub depth: usize,
    pub lower_bound: f64,
    pub incumbent: f64,
    pub fractional: PrimalHeuristicSolution,
    /// Present only for enumerated-pattern callbacks; original route identities
    /// are preserved and patterns are ordered by current reduced cost.
    pub patterns: Option<Vec<NetworkRoute>>,
    /// None denotes full coverage; a finite cutoff covers only improving points
    /// below that objective. A pattern list alone never certifies completeness.
    pub coverage_cutoff: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimalHeuristicSource {
    FractionalSolution,
    EnumeratedPatterns,
}

type Callback =
    dyn Fn(&PrimalHeuristicInput) -> Result<Option<PrimalHeuristicSolution>> + Send + Sync;

/// Runtime callback registration; cloneable across independent tree workers.
#[derive(Clone)]
pub struct PrimalHeuristic {
    pub name: String,
    pub source: PrimalHeuristicSource,
    callback: Arc<Callback>,
}
impl std::fmt::Debug for PrimalHeuristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimalHeuristic")
            .field("name", &self.name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl PrimalHeuristic {
    /// Register a callback on the current original fractional master solution.
    pub fn fractional(
        name: impl Into<String>,
        callback: impl Fn(&PrimalHeuristicInput) -> Result<Option<PrimalHeuristicSolution>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            source: PrimalHeuristicSource::FractionalSolution,
            callback: Arc::new(callback),
        }
    }
    /// Register a callback receiving a retained enumerated pricing domain.
    /// It is not invoked when no enumerated domain is available.
    pub fn enumerated(
        name: impl Into<String>,
        callback: impl Fn(&PrimalHeuristicInput) -> Result<Option<PrimalHeuristicSolution>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            source: PrimalHeuristicSource::EnumeratedPatterns,
            callback: Arc::new(callback),
        }
    }
}

fn fractional<const D: usize>(output: &Output<D>) -> PrimalHeuristicSolution {
    PrimalHeuristicSolution {
        routes: output
            .routes
            .iter()
            .zip(&output.values)
            .filter(|(_, q)| **q != 0.)
            .map(|(route, &quantity)| FixedRoute {
                route: route.clone(),
                quantity,
            })
            .chain(output.fixed_routes.iter().cloned())
            .collect(),
        master_values: output.master_values.clone(),
    }
}

fn patterns<const D: usize>(output: &Output<D>) -> Result<Option<Vec<NetworkRoute>>> {
    let state = &output.snapshot.state;
    let tags = if let Some(tags) = &state.universe {
        tags.clone()
    } else if state.gap_pool_cutoff.is_some() {
        state
            .routes
            .iter()
            .enumerate()
            .filter_map(|(tag, route)| {
                state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network)
                    .filter(|o| o.is_enumerated(&route.route))
                    .map(|_| tag)
            })
            .collect()
    } else {
        return Ok(None);
    };
    let mut paths = Vec::with_capacity(tags.len());
    for tag in tags {
        let route = state.routes[tag].as_ref();
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let mut reduced = original_costs::route_cost(oracle, &route.route)?;
        for (row, coefficient) in entries(oracle, &route.route, state.shared_offset)? {
            reduced -= output.duals[row] * coefficient;
        }
        paths.push((reduced, route.clone()));
    }
    paths.sort_by(|(a, ar), (b, br)| {
        a.total_cmp(b)
            .then(ar.network.cmp(&br.network))
            .then_with(|| ar.route.identity().cmp(&br.route.identity()))
    });
    Ok(Some(paths.into_iter().map(|(_, route)| route).collect()))
}

fn validate<const D: usize>(
    node: &NodeSnapshot<D>,
    mut solution: PrimalHeuristicSolution,
) -> Result<Option<IntegerWitness>> {
    for fixed in &mut solution.routes {
        if !fixed.quantity.is_finite()
            || fixed.quantity <= 0.
            || fixed.quantity.fract() != 0.
            || fixed.quantity > 9_007_199_254_740_992.
        {
            return Ok(None);
        }
        let Some(oracle) = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == fixed.route.network)
        else {
            return Ok(None);
        };
        let mut nodes = vec![oracle.network.graph.source as i32];
        let mut tail = oracle.network.graph.source;
        for id in &fixed.route.route.arcs {
            let Some(&index) = oracle.index.get(id) else {
                return Ok(None);
            };
            let arc = &oracle.network.graph.arcs[index];
            if arc.tail != tail {
                return Ok(None);
            }
            tail = arc.head;
            nodes.push(tail as i32);
        }
        if tail != oracle.network.graph.sink {
            return Ok(None);
        }
        fixed.route.route.nodes = nodes;
    }
    let mut provided = HashMap::new();
    for (id, value) in solution.master_values {
        if !value.is_finite() || provided.insert(id, value).is_some() {
            return Ok(None);
        }
    }
    let mut values = vec![];
    for variable in &node.state.master_variables {
        values.push((variable.id, provided.remove(&variable.id).unwrap_or(0.)));
    }
    if !provided.is_empty() {
        return Ok(None);
    }
    let Some(scheduled) = synchronization::witness(&node.state, &solution.routes)? else {
        return Ok(None);
    };
    solution.routes = scheduled;
    // Recompute reported costs as well as the certified upward objective.
    for fixed in &mut solution.routes {
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == fixed.route.network)
            .unwrap();
        if !lifecycle::integer_route_feasible(oracle, &fixed.route.route)? {
            return Ok(None);
        }
        fixed.route.route.original_cost = original_costs::route_cost(oracle, &fixed.route.route)?;
        fixed.route.route.cost = fixed.route.route.original_cost;
    }
    Ok(validate_routes(&node.state, &solution.routes, &mut values)?
        .map(|cost| (cost, solution.routes, values)))
}

pub(super) fn run<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    result: &mut ModelTreeOutput,
    deadline: Instant,
) -> Result<()> {
    if options.primal_heuristics.is_empty() {
        return Ok(());
    }
    let mut input = PrimalHeuristicInput {
        depth: output.snapshot.node_depth(),
        lower_bound: output.master.bound,
        incumbent: result.incumbent,
        fractional: fractional(output),
        patterns: None,
        coverage_cutoff: output.snapshot.state.gap_pool_cutoff,
    };
    let pool = if options
        .primal_heuristics
        .iter()
        .any(|h| h.source == PrimalHeuristicSource::EnumeratedPatterns)
    {
        patterns(output)?
    } else {
        None
    };
    for heuristic in &options.primal_heuristics {
        if Instant::now() >= deadline {
            break;
        }
        input.patterns = match heuristic.source {
            PrimalHeuristicSource::FractionalSolution => None,
            PrimalHeuristicSource::EnumeratedPatterns => {
                let Some(pool) = &pool else {
                    continue;
                };
                Some(pool.clone())
            }
        };
        input.incumbent = result.incumbent;
        if let Some(solution) = (heuristic.callback)(&input)? {
            if let Some((objective, routes, values)) = validate(&output.snapshot, solution)? {
                if objective < result.incumbent {
                    result.incumbent = objective;
                    result.routes = routes;
                    result.master_values = values;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/primal_callback_tests.rs"]
mod tests;
