use super::*;
use crate::master::LpMaster;

/// Coefficients of one additive shared row for one pricing network.
/// Arc coefficients follow the original graph's physical arc order.
#[derive(Clone, Debug)]
pub struct RowCoefficients {
    pub network: i32,
    pub arcs: Vec<f64>,
    pub per_route: f64,
}

#[derive(Clone, Debug)]
pub struct FixedRoute {
    pub route: NetworkRoute,
    pub quantity: f64,
}

/// Fractional use of a unit-cover shared row by one network.
#[derive(Clone, Copy, Debug)]
pub struct Assignment {
    pub shared_row: usize,
    pub network: i32,
    pub value: f64,
}

/// Owned master, route registry and independent pricing states for every network.
/// Solving, branching and fixing fork this state; the parent remains reusable.
pub struct NodeSnapshot<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) master: ClpMaster,
    pub(super) warm: cg::WarmStart,
    pub(super) state: State<D>,
}

impl<const D: usize> std::fmt::Debug for NodeSnapshot<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetworkNodeSnapshot")
            .field("networks", &self.state.oracles.len())
            .field("rows", &self.master.rhs.len())
            .field("columns", &self.master.costs.len())
            .field("fixed_routes", &self.state.fixed_routes.len())
            .finish()
    }
}

impl<const D: usize> NodeSnapshot<D> {
    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self {
            master: self.master.try_clone()?,
            warm: self.warm.clone(),
            state: State {
                oracles: self
                    .state
                    .oracles
                    .iter()
                    .map(|o| Oracle {
                        network: o.network.clone(),
                        total_multiplicity: o.total_multiplicity,
                        solver: o.solver.fork_for_node(),
                        index: o.index.clone(),
                        costs: vec![],
                        route_entries: o.route_entries.clone(),
                        calls: 0,
                        nonlinear_rows: o.nonlinear_rows.clone(),
                        knapsack_rows: o.knapsack_rows.clone(),
                        presence_rows: o.presence_rows.clone(),
                        resource_groups: o.resource_groups.clone(),
                        proved_zero_arcs: o.proved_zero_arcs.clone(),
                        proved_multiplicity_upper: o.proved_multiplicity_upper,
                    })
                    .collect(),
                shared_offset: self.state.shared_offset,
                routes: self.state.routes.clone(),
                tags: self.state.tags.clone(),
                correction: None,
                gradient: None,
                rhs: self.state.rhs.clone(),
                deadline: self.state.deadline,
                shared_rows: self.state.shared_rows.clone(),
                fixed_routes: self.state.fixed_routes.clone(),
                universe: self.state.universe.clone(),
                enumeration_report: self.state.enumeration_report.clone(),
                preprocessing: self.state.preprocessing.clone(),
                master_variables: self.state.master_variables.clone(),
                route_upper_bounds: self.state.route_upper_bounds.clone(),
            },
        })
    }

    /// Resume this node with a new time/iteration budget and no external registry.
    pub fn solve(&self, mut options: Options) -> Result<Output<D>> {
        validate_options(&options)?;
        let deadline = Instant::now()
            .checked_add(options.cg.time_limit)
            .ok_or_else(|| invalid("network time limit"))?;
        let mut node = self.fork()?;
        node.state.deadline = deadline;
        options.cg.warm_start = Some(node.warm);
        run(node.master, node.state, options)
    }

    /// Append a shared additive row to both retained columns and future pricing.
    /// Omitted networks have zero coefficients. The caller supplies a valid cut
    /// or branch restriction; this method does not infer cut validity.
    pub fn with_row(&self, row: Row, coefficients: &[RowCoefficients]) -> Result<Self> {
        if !row.rhs.is_finite() {
            return Err(invalid("network row RHS"));
        }
        let mut seen = HashSet::new();
        for c in coefficients {
            let oracle = self
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == c.network)
                .ok_or_else(|| invalid("unknown row network"))?;
            if !seen.insert(c.network)
                || c.arcs.len() != oracle.network.graph.arcs.len()
                || !c.per_route.is_finite()
                || c.arcs.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("network row coefficients"));
            }
        }
        let mut node = self.fork()?;
        let shared = node.state.shared_rows.len();
        let physical = node.master.rhs.len();
        for oracle in &mut node.state.oracles {
            if let Some(c) = coefficients.iter().find(|c| c.network == oracle.network.id) {
                if c.arcs.iter().any(|&value| value != 0.) {
                    for (entries, &value) in Arc::make_mut(&mut oracle.network.arc_entries)
                        .iter_mut()
                        .zip(&c.arcs)
                    {
                        if value != 0. {
                            entries.push((shared, value));
                        }
                    }
                }
                if c.per_route != 0. {
                    Arc::make_mut(&mut oracle.network.route_entries).push((shared, c.per_route));
                }
            }
        }
        let route_value = |route: &NetworkRoute| -> Result<f64> {
            let oracle = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == route.network)
                .unwrap();
            Ok(entries(oracle, &route.route, node.state.shared_offset)?
                .into_iter()
                .find(|&(r, _)| r == physical)
                .map_or(0., |(_, v)| v))
        };
        let mut rhs = row.rhs;
        for fixed in &node.state.fixed_routes {
            rhs -= fixed.quantity * route_value(&fixed.route)?;
        }
        if !rhs.is_finite() {
            return Err(invalid("fixed network row overflow"));
        }
        let mut column_entries = vec![];
        for (column, metadata) in node.warm.columns.iter_mut().enumerate() {
            if let Some(metadata) = metadata {
                if metadata.tag < 0 {
                    continue;
                }
                let value = route_value(&node.state.routes[metadata.tag as usize])?;
                if value != 0. {
                    column_entries.push((column, value));
                    metadata.entries.push((physical, value));
                }
            }
        }
        for metadata in &mut node.warm.generated_pool {
            if metadata.tag < 0 {
                continue;
            }
            let value = route_value(&node.state.routes[metadata.tag as usize])?;
            if value != 0. {
                metadata.entries.push((physical, value));
            }
        }
        node.master.add_row(row.sense, rhs, &column_entries)?;
        let sense = row.sense;
        node.state.shared_rows.push(row);
        node.state.rhs = node.master.rhs.clone();
        // Row dimensions and objectives differ; the old smoothing center is
        // not a dual point for this child. Keep valid inherited lower bounds.
        node.warm.stabilization = None;
        let penalty = node
            .warm
            .artificials
            .iter()
            .map(|&c| node.master.costs[c])
            .fold(2.76_f64, f64::max);
        for sign in [1., -1.] {
            if (sign > 0. && sense == RowSense::LessEqual)
                || (sign < 0. && sense == RowSense::GreaterEqual)
            {
                continue;
            }
            let column = node.master.add_column(penalty, &[(physical, sign)])?;
            node.warm.columns.push(None);
            node.warm.artificials.push(column);
        }
        Ok(node)
    }

    fn validate_assignment(&self, shared_row: usize, network: i32) -> Result<()> {
        let row = self
            .state
            .shared_rows
            .get(shared_row)
            .ok_or_else(|| invalid("unknown assignment shared row"))?;
        if self.state.oracles.iter().any(|o| {
            o.nonlinear_rows
                .iter()
                .chain(&o.knapsack_rows)
                .chain(&o.presence_rows)
                .any(|&(_, r)| r == shared_row)
        }) || row.sense != RowSense::Equal
            || row.rhs != 1.
            || !self.state.oracles.iter().any(|o| o.network.id == network)
        {
            return Err(invalid(
                "assignment requires a unit-cover row and known network",
            ));
        }
        // Integer route quantities must imply integer, nonnegative assignment
        // quantities. Arbitrary signed/shared linear rows are not packing sets.
        for oracle in &self.state.oracles {
            if oracle
                .network
                .arc_entries
                .iter()
                .flatten()
                .chain(oracle.network.route_entries.iter())
                .any(|&(r, v)| r == shared_row && (v < 0. || v.fract() != 0.))
            {
                return Err(invalid(
                    "assignment requires nonnegative integer coefficients",
                ));
            }
        }
        Ok(())
    }

    /// Enforce use >= 1 (assigned) or use <= 0 (excluded) by one network.
    /// Membership uses shared-row coefficients, never local vertex/arc IDs.
    pub fn branch_assignment(
        &self,
        shared_row: usize,
        network: i32,
        assigned: bool,
    ) -> Result<Self> {
        self.validate_assignment(shared_row, network)?;
        let oracle = self
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == network)
            .unwrap();
        let sum = |values: &[(usize, f64)]| {
            values
                .iter()
                .filter(|&&(r, _)| r == shared_row)
                .map(|&(_, v)| v)
                .sum()
        };
        let coefficients = RowCoefficients {
            network,
            arcs: oracle.network.arc_entries.iter().map(|v| sum(v)).collect(),
            per_route: sum(&oracle.network.route_entries),
        };
        let (sense, rhs) = if assigned {
            (RowSense::GreaterEqual, 1.)
        } else {
            (RowSense::LessEqual, 0.)
        };
        self.with_row(Row { sense, rhs }, &[coefficients])
    }

    /// Candidates from the current solved, original-feasible node only.
    pub fn fractional_assignments(&self, tolerance: f64) -> Result<Vec<Assignment>> {
        if !tolerance.is_finite() || tolerance <= 0. || tolerance >= 0.5 {
            return Err(invalid(
                "assignment tolerance must be between zero and one half",
            ));
        }
        if !self.master.solved()
            || self
                .warm
                .artificials
                .iter()
                .any(|&c| self.master.values()[c].abs() > tolerance)
        {
            return Err(invalid(
                "assignment candidates require an original-feasible solved node",
            ));
        }
        let mut candidates = vec![];
        for shared_row in 0..self.state.shared_rows.len() {
            for oracle in &self.state.oracles {
                if self
                    .validate_assignment(shared_row, oracle.network.id)
                    .is_err()
                {
                    continue;
                }
                let physical = self.state.shared_offset + shared_row;
                let coefficient = |route: &NetworkRoute| -> Result<f64> {
                    if route.network != oracle.network.id {
                        return Ok(0.);
                    }
                    Ok(entries(oracle, &route.route, self.state.shared_offset)?
                        .into_iter()
                        .find(|&(r, _)| r == physical)
                        .map_or(0., |(_, v)| v))
                };
                let mut value = 0.;
                for (column, metadata) in self.warm.columns.iter().enumerate() {
                    if let Some(metadata) = metadata {
                        if metadata.tag < 0 {
                            continue;
                        }
                        value += self.master.values()[column]
                            * coefficient(&self.state.routes[metadata.tag as usize])?;
                    }
                }
                for fixed in &self.state.fixed_routes {
                    value += fixed.quantity * coefficient(&fixed.route)?;
                }
                if value > tolerance && value < 1. - tolerance {
                    candidates.push(Assignment {
                        shared_row,
                        network: oracle.network.id,
                        value,
                    });
                }
            }
        }
        Ok(candidates)
    }

    /// Fix additional integer copies of a registered route. Every row (including
    /// cuts and branches), network multiplicity and objective offset is adjusted.
    pub fn fix_route(&self, route: usize, quantity: f64) -> Result<Self> {
        let route = self
            .state
            .routes
            .get(route)
            .ok_or_else(|| invalid("unknown fixed route"))?;
        if !quantity.is_finite()
            || quantity <= 0.
            || quantity.fract() != 0.
            || quantity > 9_007_199_254_740_992.
        {
            return Err(invalid("fixed route quantity must be a positive integer"));
        }
        let capped = if let Some(universe) = &self.state.universe {
            let canonical = universe
                .iter()
                .copied()
                .find(|&tag| {
                    let candidate = &self.state.routes[tag];
                    candidate.network == route.network && candidate.route.arcs == route.route.arcs
                })
                .ok_or_else(|| invalid("fixed route absent from the active complete domain"))?;
            let upper = enumeration::route_upper(&self.state, canonical)
                .ok_or_else(|| invalid("fixed route cap"))?;
            if quantity > upper {
                return Err(invalid("fixed quantity exceeds remaining route cap"));
            }
            if upper.is_finite() {
                let remaining = -crate::core::rounding::add_down(quantity, -upper)
                    .ok_or_else(|| invalid("fixed route cap arithmetic"))?;
                Some((canonical, remaining.max(0.)))
            } else {
                None
            }
        } else {
            None
        };
        let mut node = self.fork()?;
        let oracle = node
            .state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == route.network)
            .unwrap();
        if quantity > oracle.multiplicity_upper() {
            return Err(invalid("fixed route exceeds remaining multiplicity"));
        }
        if !integer_route_feasible(oracle, &route.route)? {
            return Err(invalid(
                "fixed route must be an original-feasible elementary path",
            ));
        }
        let coefficients = entries(oracle, &route.route, node.state.shared_offset)?;
        let mut rhs = node.master.rhs.clone();
        for (row, value) in coefficients {
            rhs[row] -= quantity * value;
        }
        let offset = node.master.objective_offset + quantity * route.route.original_cost;
        if !offset.is_finite() || rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("fixed network contribution overflow"));
        }
        oracle.network.multiplicity.upper -= quantity;
        if let Some(upper) = oracle.proved_multiplicity_upper {
            oracle.proved_multiplicity_upper = Some(
                crate::core::rounding::add_up(upper, -quantity)
                    .ok_or_else(|| invalid("fixed multiplicity proof arithmetic"))?
                    .max(0.),
            );
        }
        oracle.network.multiplicity.lower = (oracle.network.multiplicity.lower - quantity).max(0.);
        if let Some((tag, upper)) = capped {
            node.state.route_upper_bounds.insert(tag, upper);
            for (column, metadata) in node.warm.columns.iter().enumerate() {
                if metadata.as_ref().is_some_and(|m| m.tag == tag as i32)
                    && upper < node.master.upper[column]
                {
                    node.master.set_upper(column, upper)?;
                }
            }
        }
        node.master.set_rhs(&rhs)?;
        node.master.objective_offset = offset;
        node.state.rhs = rhs;
        node.state.fixed_routes.push(FixedRoute {
            route: route.as_ref().clone(),
            quantity,
        });
        node.warm.stabilization = None;
        Ok(node)
    }
}

// Reuse the routing validator's exact-real resource replay rather than trusting
// the rounded resource profile returned by labeling.
pub(super) fn integer_route_feasible<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<bool> {
    let problem = crate::master::root::RootProblem::set_partitioning(
        super::resource_groups::domain_graph(oracle)?.into_owned(),
        1.,
    )?;
    Ok(
        crate::master::root::exact_route_feasible(&problem, route).is_some()
            && oracle
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?,
    )
}

/// Fixed reductions replay original contributions even when a later branch
/// contradicts the fixed schedule. That contradiction can prove infeasibility;
/// it must not be mistaken for an inexact arithmetic reduction.
pub(super) fn original_route_feasible<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<bool> {
    let problem = crate::master::root::RootProblem::set_partitioning(
        oracle.network.graph.as_ref().clone(),
        1.,
    )?;
    Ok(
        crate::master::root::exact_route_feasible(&problem, route).is_some()
            && oracle
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?,
    )
}

/// Replay explicit fixed copies into a fresh model before preprocessing or pricing.
/// Only snapshot-owned, originally validated physical routes enter this path.
pub(super) fn install_fixed_routes<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    fixed: &[FixedRoute],
) -> Result<()> {
    for fixed in fixed {
        let oracle = state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == fixed.route.network)
            .ok_or_else(|| invalid("unknown fixed route network"))?;
        let q = fixed.quantity;
        if !q.is_finite()
            || q <= 0.
            || q.fract() != 0.
            || q > 9_007_199_254_740_992.
            || q > oracle.network.multiplicity.upper
            || !integer_route_feasible(oracle, &fixed.route.route)?
        {
            return Err(invalid("invalid replayed fixed route"));
        }
        for (row, value) in entries(oracle, &fixed.route.route, state.shared_offset)? {
            state.rhs[row] -= q * value;
        }
        master.objective_offset += q * fixed.route.route.original_cost;
        if !master.objective_offset.is_finite() || state.rhs.iter().any(|v| !v.is_finite()) {
            return Err(invalid("fixed network contribution overflow"));
        }
        oracle.network.multiplicity.upper -= q;
        oracle.network.multiplicity.lower = (oracle.network.multiplicity.lower - q).max(0.);
        state.fixed_routes.push(fixed.clone());
    }
    master.set_rhs(&state.rhs)?;
    Ok(())
}

/// Transfer internally generated phase-I paths into a fresh objective master.
/// Identities are rebuilt and costs replayed from the original physical graph.
pub(super) fn install_seed_routes<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    known: &mut Vec<cg::Column>,
    seeds: &[NetworkRoute],
) -> Result<()> {
    for seed in seeds {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == seed.network)
            .ok_or_else(|| invalid("unknown seed route network"))?;
        let identity = seed.route.identity();
        let key = (seed.network, identity.clone());
        if state.tags.contains_key(&key) {
            continue;
        }
        let coefficients = entries(oracle, &seed.route, state.shared_offset)?;
        let mut arcs = seed.route.arcs.clone();
        arcs.sort_unstable();
        let cost = arcs
            .iter()
            .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
            .sum::<f64>()
            + oracle
                .network
                .graph
                .coupled_path_cost(
                    &seed
                        .route
                        .arcs
                        .iter()
                        .map(|id| oracle.index[id])
                        .collect::<Vec<_>>(),
                )?
                .ok_or_else(|| invalid("seed path violates coupled resources"))?;
        if !cost.is_finite() {
            return Err(invalid("seed route cost overflow"));
        }
        let tag =
            i32::try_from(state.routes.len()).map_err(|_| invalid("network route tag overflow"))?;
        let mut route = seed.clone();
        route.route.original_cost = cost;
        route.route.cost = cost;
        let column = master.add_column(cost, &coefficients)?;
        debug_assert_eq!(column, known.len());
        known.push(cg::Column {
            cost,
            reduced_cost: 0.,
            entries: coefficients,
            acceptable: true,
            tag,
            identity: format!("{}:{identity}", seed.network),
        });
        state.tags.insert(key, tag);
        state.routes.push(Arc::new(route));
    }
    Ok(())
}
