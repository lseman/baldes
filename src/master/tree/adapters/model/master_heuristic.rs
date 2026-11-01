//! Residual user master heuristic before dive cuts and column generation.
use super::*;

/// Read-only child state before reoptimization. The route vector mirrors the
/// generated-column LP input; fixed contributions are exposed separately.
#[derive(Clone, Debug)]
pub struct MasterHeuristicInput {
    pub routes: Vec<FixedRoute>,
    pub fixed_routes: Vec<FixedRoute>,
    pub master_variables: Vec<ResidualMasterCoordinate>,
}

/// An original static coordinate expressed as `original = offset + residual`.
/// Bounds include explicit child restrictions, translated into residual space.
#[derive(Clone, Debug)]
pub struct ResidualMasterCoordinate {
    pub id: i32,
    /// Original objective coefficient, unchanged by residual translation.
    pub cost: f64,
    pub integer: bool,
    pub offset: f64,
    pub bounds: crate::master::preprocessing::Bounds,
    /// Retained parent LP value in residual coordinates. It may violate this
    /// child's bounds because the child has not reoptimized yet.
    pub value: f64,
}

/// A residual master candidate. Fixed routes and static offsets are restored
/// internally. Omitted residual static values are zero. Route and static
/// quantities may be fractional; incumbents must satisfy declared integrality.
#[derive(Clone, Debug, Default)]
pub struct ResidualMasterSolution {
    pub routes: Vec<FixedRoute>,
    pub master_values: Vec<(i32, f64)>,
}

type Callback =
    dyn Fn(&MasterHeuristicInput) -> Result<Option<ResidualMasterSolution>> + Send + Sync;

/// Single application master heuristic, disabled unless registered. Runs on
/// ordinary and local-search dive children, before fixed-solution separators;
/// it is independent of post-CG primal heuristics and the greedy evaluator.
#[derive(Clone)]
pub struct MasterHeuristic {
    pub name: String,
    callback: Arc<Callback>,
}
impl std::fmt::Debug for MasterHeuristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MasterHeuristic")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
impl MasterHeuristic {
    pub fn new(
        name: impl Into<String>,
        callback: impl Fn(&MasterHeuristicInput) -> Result<Option<ResidualMasterSolution>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            callback: Arc::new(callback),
        }
    }
}

pub(super) struct Prepared<const D: usize> {
    pub node: Option<NodeSnapshot<D>>,
    /// None retains the LP map; Some replaces it, including the empty map.
    pub routes: Option<Vec<FixedRoute>>,
}

fn coalesce(routes: impl IntoIterator<Item = FixedRoute>) -> Vec<FixedRoute> {
    let mut result: Vec<FixedRoute> = Vec::new();
    for route in routes {
        if let Some(old) = result.iter_mut().find(|old| {
            old.route.network == route.route.network
                && old.route.route.identity() == route.route.route.identity()
        }) {
            old.quantity += route.quantity;
        } else {
            result.push(route);
        }
    }
    result
}

pub(super) fn retained_witness<const D: usize>(
    node: &NodeSnapshot<D>,
) -> Result<Option<IntegerWitness>> {
    let Some(witness) = &node.state.pricing_witness else {
        return Ok(None);
    };
    let mut values = witness.2.clone();
    Ok(validate_routes(&node.state, &witness.1, &mut values)?
        .map(|objective| (objective, witness.1.clone(), values)))
}

// The same private registry/backend/warm-start installation used by generated
// seed columns. Semantic entries must be projected past retired physical rows.
fn load<const D: usize>(node: &mut NodeSnapshot<D>, routes: &[FixedRoute]) -> Result<bool> {
    for fixed in routes {
        let route = &fixed.route;
        let identity = route.route.identity();
        let key = (route.network, identity.clone());
        let oracle = node
            .state
            .oracles
            .iter()
            .position(|o| o.network.id == route.network)
            .ok_or_else(|| invalid("master heuristic route network"))?;
        let tag = if let Some(&tag) = node.state.tags.get(&key) {
            tag
        } else {
            let tag = i32::try_from(node.state.routes.len())
                .map_err(|_| invalid("master heuristic route tag overflow"))?;
            Arc::make_mut(&mut node.state.routes).push(Arc::new(route.clone()));
            Arc::make_mut(&mut node.state.tags).insert(key, tag);
            tag
        };
        let mut cap = node.state.oracles[oracle].multiplicity_upper().min(
            node.state
                .route_upper_bounds
                .get(&(tag as usize))
                .copied()
                .unwrap_or(f64::INFINITY),
        );
        if let Some(column) = node
            .warm
            .columns
            .iter()
            .position(|metadata| metadata.as_ref().is_some_and(|c| c.tag == tag))
        {
            cap = cap.min(node.master.upper[column]);
        } else {
            let entries =
                retirement::active_entries(&node.state, &node.state.oracles[oracle], &route.route)?;
            let metadata = cg::Column {
                cost: route.route.original_cost,
                reduced_cost: 0.,
                entries,
                acceptable: true,
                tag,
                identity: format!("{}:{identity}", route.network),
            };
            let column = node.master.add_column(metadata.cost, &metadata.entries)?;
            debug_assert_eq!(column, node.warm.columns.len());
            node.master.set_upper(column, cap.min(f64::MAX))?;
            node.warm.columns.push(Some(metadata.clone()));
            node.warm.generated_pool.retain(|known| known.tag != tag);
            node.warm.generated_pool.push(metadata);
        }
        if fixed.quantity > cap {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn prepare<const D: usize>(
    node: &NodeSnapshot<D>,
    options: &ModelTreeOptions,
    residuals: &HashMap<usize, f64>,
) -> Result<Prepared<D>> {
    let Some(heuristic) = &options.master_heuristic else {
        return Ok(Prepared {
            node: None,
            routes: None,
        });
    };
    let mut master_values = vec![0.; node.state.master_variables.len()];
    for (column, metadata) in node.warm.columns.iter().enumerate() {
        if let Some((index, sign)) = metadata
            .as_ref()
            .and_then(|c| master_variables::decode(c.tag))
        {
            if let Some(value) = master_values.get_mut(index) {
                *value += sign * node.master.values().get(column).copied().unwrap_or(0.);
            }
        }
    }
    let input = MasterHeuristicInput {
        routes: node
            .warm
            .columns
            .iter()
            .enumerate()
            .filter_map(|(column, metadata)| {
                let tag = usize::try_from(metadata.as_ref()?.tag).ok()?;
                let route = node.state.routes.get(tag)?;
                let quantity = *node.master.values().get(column)?;
                (quantity != 0.).then(|| FixedRoute {
                    route: route.as_ref().clone(),
                    quantity,
                })
            })
            .collect(),
        fixed_routes: coalesce(node.state.fixed_routes.clone()),
        master_variables: node
            .state
            .master_variables
            .iter()
            .enumerate()
            .map(|(index, variable)| {
                let offset = residuals.get(&index).copied().unwrap_or(0.);
                let value = master_values[index] - offset;
                ResidualMasterCoordinate {
                    id: variable.id,
                    cost: variable.cost,
                    integer: variable.integer,
                    offset,
                    bounds: crate::master::preprocessing::Bounds {
                        lower: variable.bounds.lower - offset,
                        upper: variable.bounds.upper - offset,
                    },
                    value,
                }
            })
            .collect(),
    };
    // The native lifecycle clears its current primal map before invoking the
    // functor. A declined or invalid candidate leaves no residual map for cuts.
    let empty = || Prepared {
        node: None,
        routes: Some(vec![]),
    };
    let Some(mut candidate) = (heuristic.callback)(&input)? else {
        return Ok(empty());
    };
    let residual_count = candidate.routes.len();
    candidate
        .routes
        .extend(node.state.fixed_routes.iter().cloned());
    let mut seen = HashSet::new();
    for (id, value) in &mut candidate.master_values {
        if !seen.insert(*id) {
            return Ok(empty());
        }
        if let Some(index) = node.state.master_variables.iter().position(|v| v.id == *id) {
            *value += residuals.get(&index).copied().unwrap_or(0.);
        }
    }
    for (&index, &offset) in residuals {
        let id = node.state.master_variables[index].id;
        if !seen.contains(&id) {
            candidate.master_values.push((id, offset));
        }
    }
    let mut child = node.fork()?;
    let integral: Vec<_> = child
        .state
        .master_variables
        .iter()
        .map(|v| v.integer)
        .collect();
    for variable in &mut child.state.master_variables {
        variable.integer = false;
    }
    let validated = primal_callbacks::validate_point(
        &child,
        PrimalHeuristicSolution {
            routes: candidate.routes,
            master_values: candidate.master_values,
        },
        false,
    );
    for (variable, integer) in child.state.master_variables.iter_mut().zip(integral) {
        variable.integer = integer;
    }
    let Some(mut witness) = validated? else {
        return Ok(empty());
    };
    let residual_routes = coalesce(witness.1.iter().take(residual_count).cloned());
    if !load(&mut child, &residual_routes)? {
        return Ok(empty());
    }
    witness.1 = coalesce(witness.1);
    // Original integrality remains authoritative for incumbent publication.
    if witness.1.iter().all(|route| route.quantity.fract() == 0.) {
        if let Some(objective) = validate_routes(&child.state, &witness.1, &mut witness.2)? {
            witness.0 = objective;
            if child
                .state
                .pricing_witness
                .as_ref()
                .map_or(true, |old| old.0 > witness.0)
            {
                child.state.pricing_witness = Some(witness);
            }
        }
    }
    Ok(Prepared {
        node: Some(child),
        routes: Some(residual_routes),
    })
}
