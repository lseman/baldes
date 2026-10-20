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
    pub(super) fn fork(&self) -> Result<Self> {
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
                        route_entries: o.route_entries.clone(),
                        calls: 0,
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
                for (entries, &value) in oracle.network.arc_entries.iter_mut().zip(&c.arcs) {
                    if value != 0. {
                        entries.push((shared, value));
                    }
                }
                if c.per_route != 0. {
                    oracle.network.route_entries.push((shared, c.per_route));
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
                let value = route_value(&node.state.routes[metadata.tag as usize])?;
                if value != 0. {
                    column_entries.push((column, value));
                    metadata.entries.push((physical, value));
                }
            }
        }
        for metadata in &mut node.warm.generated_pool {
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
        if row.sense != RowSense::Equal
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
                .chain(&oracle.network.route_entries)
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
        let mut node = self.fork()?;
        let oracle = node
            .state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == route.network)
            .unwrap();
        if quantity > oracle.network.multiplicity.upper {
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
        oracle.network.multiplicity.lower = (oracle.network.multiplicity.lower - quantity).max(0.);
        node.master.set_rhs(&rhs)?;
        node.master.objective_offset = offset;
        node.state.rhs = rhs;
        node.state.fixed_routes.push(FixedRoute {
            route: route.clone(),
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
        oracle.network.graph.clone(),
        oracle.total_multiplicity.upper.max(1.),
    )?;
    Ok(
        crate::master::root::exact_route_feasible(&problem, route).is_some()
            && oracle
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?,
    )
}
