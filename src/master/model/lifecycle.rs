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

fn appended_row_value<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    offset: usize,
    shared: usize,
    physical: usize,
    coefficient: Option<&RowCoefficients>,
) -> Result<f64> {
    // Rich replay retains its validation and coefficient staging. Plain retained
    // columns already validated every older row; only this association is new.
    if offset.checked_add(shared) != Some(physical)
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.activity_rows.is_empty()
        || !oracle.resource_groups.is_empty()
        || !oracle.ryan_foster_rows.is_empty()
        || !oracle.arc_set_branch_rows.is_empty()
    {
        return Ok(entries(oracle, route, offset)?
            .into_iter()
            .find(|&(row, _)| row == physical)
            .map_or(0., |(_, value)| value));
    }
    let mut value = 0.;
    if let Some(c) = coefficient {
        if c.per_route != 0. {
            value += c.per_route;
        }
    }
    for id in &route.arcs {
        let index = *oracle
            .index
            .get(id)
            .ok_or_else(|| invalid("unknown network route arc"))?;
        if let Some(c) = coefficient {
            let contribution = c.arcs[index];
            if contribution != 0. {
                value += contribution;
            }
        }
    }
    if !value.is_finite() {
        return Err(invalid("network route coefficient overflow"));
    }
    // Full replay drops zero coefficients; its caller supplies positive zero.
    Ok(if value == 0. { 0. } else { value })
}

/// Owned master, route registry and independent pricing states for every network.
/// Solving, branching and fixing fork this state; the parent remains reusable.
pub struct NodeSnapshot<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) master: ClpMaster,
    pub(super) warm: RetainedWarmStart,
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

impl<const D: usize> State<D> {
    pub(super) fn fork_state(&self) -> Self {
        State {
            component_branches: self.component_branches.clone(),
            component_count_bounds: self.component_count_bounds.clone(),
            component_local_rows: self.component_local_rows.clone(),
            component_branch_exclusions: self.component_branch_exclusions.clone(),
            pricing_witness: self.pricing_witness.clone(),
            tree_incumbent: self.tree_incumbent,
            integer_primal_termination: self.integer_primal_termination,
            witness_tolerance: self.witness_tolerance,
            box_pricing_admission: self.box_pricing_admission,
            ordinary_artificial_cost: self.ordinary_artificial_cost,
            cut_artificial_bases: self.cut_artificial_bases.clone(),
            variable_boxes: self.variable_boxes.clone(),
            grouped_variable_boxes: self.grouped_variable_boxes.clone(),
            original_variable_costs: self.original_variable_costs.clone(),
            pricing_master: None,
            pricing_duals: vec![],
            pricing_rhs: vec![],
            pricing_order_scratch: vec![],
            pricing_gap: self.pricing_gap,
            previous_fixing_gap: self.previous_fixing_gap,
            gap_enumeration: self.gap_enumeration,
            tree_bound_floor: self.tree_bound_floor,
            fixing_bound: None,
            fixing_cutoffs: None,
            gap_checkpoint: None,
            enumerated_routes: self.enumerated_routes,
            gap_pool_cutoff: self.gap_pool_cutoff,
            gap_mip: self.gap_mip.clone(),
            fixing_calls: self.fixing_calls,
            objective_fixed: self.objective_fixed,
            phase_one_infeasible: self.phase_one_infeasible,
            mapped_subtree_size: self.mapped_subtree_size,
            mapped_subtree_priority: self.mapped_subtree_priority,
            mapped_lineage: self.mapped_lineage,
            branch_history_origin: self.branch_history_origin.clone(),
            node_depth: self.node_depth,
            scheduler: self.scheduler.fresh(),
            synchronized_services: self.synchronized_services.clone(),
            oracles: self
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
                    enumerated: o.enumerated.clone(),
                    nonlinear_rows: o.nonlinear_rows.clone(),
                    knapsack_rows: o.knapsack_rows.clone(),
                    presence_rows: o.presence_rows.clone(),
                    activity_rows: o.activity_rows.clone(),
                    resource_groups: o.resource_groups.clone(),
                    ryan_foster_rows: o.ryan_foster_rows.clone(),
                    arc_set_branch_rows: o.arc_set_branch_rows.clone(),
                    synchronized_windows: o.synchronized_windows.clone(),
                    proved_zero_arcs: o.proved_zero_arcs.clone(),
                    proved_multiplicity_upper: o.proved_multiplicity_upper,
                })
                .collect(),
            shared_offset: self.shared_offset,
            routes: self.routes.clone(),
            tags: self.tags.clone(),
            correction: None,
            gradient: None,
            rhs: self.rhs.clone(),
            deadline: self.deadline,
            shared_rows: self.shared_rows.clone(),
            retired_rows: self.retired_rows.clone(),
            curvature_rows: self.curvature_rows.clone(),
            fixed_routes: self.fixed_routes.clone(),
            universe: self.universe.clone(),
            enumeration_report: self.enumeration_report.clone(),
            preprocessing: self.preprocessing.clone(),
            master_variables: self.master_variables.clone(),
            master_variable_signs: self.master_variable_signs.clone(),
            master_variable_bound_rows: self.master_variable_bound_rows.clone(),
            route_upper_bounds: self.route_upper_bounds.clone(),
            cuts: self.cuts.clone(),
        }
    }

    pub(super) fn snapshot_for_checkpoint(&self) -> Self {
        let mut forked = self.fork_state();
        for (dst, src) in forked.oracles.iter_mut().zip(&self.oracles) {
            dst.calls = src.calls;
            dst.solver.rollback_point_saved = false;
            dst.solver.requires_rollback = false;
        }
        forked
    }
}

impl<const D: usize> NodeSnapshot<D> {
    pub(crate) fn incidence_separator_history(&self) -> Option<crate::cuts::IncidenceCutSeparator> {
        self.state.cuts.as_ref()?.incidence_history()
    }

    pub(crate) fn set_incidence_separator_history(
        &mut self,
        history: &crate::cuts::IncidenceCutSeparator,
    ) {
        if let Some(cuts) = &mut self.state.cuts {
            cuts.set_incidence_history(history);
        }
    }

    /// Prepare a new tree-node evaluation or strong-branching trial. Same-node
    /// resumes deliberately bypass this reset and retain their current state.
    pub(crate) fn prepare_evaluation(&mut self) -> Result<()> {
        self.reset_evaluation_artificials()?;
        self.reset_evaluation_incidence_phase();
        Ok(())
    }

    fn reset_evaluation_incidence_phase(&mut self) {
        if self
            .state
            .cuts
            .as_ref()
            .and_then(|cuts| cuts.incidence_config())
            .is_some()
        {
            self.state.oracles[0].solver.cuts.reset_separation_phase();
        }
    }

    /// Simplex pivots recorded on this retained master since its last fork.
    /// Forks reset the counter; discarded trial/checkpoint masters are excluded.
    pub fn master_simplex_iterations(&self) -> usize {
        self.master.simplex_iterations
    }

    /// Completed IPM Newton iterations on this retained master since its last fork.
    /// Forks reset the counter; discarded trial/checkpoint masters are excluded.
    pub fn master_ipm_iterations(&self) -> usize {
        self.master.ipm_iterations
    }

    pub(crate) fn fork(&self) -> Result<Self> {
        let mut node = Self {
            master: self.master.try_clone()?,
            warm: self.warm.clone(),
            state: self.state.fork_state(),
        };
        // Incumbents are supplied by each solve's owner, never inherited as
        // authority from a reusable snapshot.
        node.state.tree_incumbent = f64::INFINITY;
        if node.state.objective_fixed {
            // The old incumbent may not be feasible for a new branch, or the
            // caller may resume this snapshot as an LP without an incumbent.
            // Drop only cost-dependent bucket fixings; graph restrictions and
            // dynamically strengthened relaxations remain in the solver.
            for oracle in &mut node.state.oracles {
                oracle.solver.restore_arcs();
            }
            node.state.objective_fixed = false;
            node.state.gap_pool_cutoff = None;
            node.warm.bound = f64::NEG_INFINITY;
            node.warm.numerical_bound = f64::NEG_INFINITY;
            node.warm.certified_bound = f64::NEG_INFINITY;
            node.warm.stabilization = None;
            node.warm.clear_proximal();
        }
        Ok(node)
    }

    /// Actual branch depth; resumes and row additions preserve this value.
    pub fn node_depth(&self) -> usize {
        self.state.node_depth
    }

    /// Start a new tree-node or trial evaluation with its original ordinary
    /// artificial costs. Same-node resumes and generic CG continuation do not
    /// call this method. Known separated-row artificials use 1.2 times their
    /// retained creation cost; columns without creation metadata stay unchanged.
    fn reset_evaluation_artificials(&mut self) -> Result<()> {
        let cost = self.state.ordinary_artificial_cost;
        let separated: HashSet<_> = self
            .warm
            .cut_artificials
            .iter()
            .map(|&(_, column)| column)
            .collect();
        let mut changed = false;
        for &column in &self.warm.artificials {
            let cost = if separated.contains(&column) {
                let Some(base) = self.state.cut_artificial_bases.get(&column) else {
                    continue;
                };
                let setup_cost = 1.2 * base;
                if !setup_cost.is_finite() || setup_cost < 0. {
                    return Err(invalid("retained cut artificial setup cost"));
                }
                setup_cost
            } else {
                cost
            };
            let previous = *self
                .master
                .costs
                .get(column)
                .ok_or_else(|| invalid("retained ordinary artificial column"))?;
            if previous != cost {
                self.master.set_cost(column, cost)?;
                changed = true;
            }
        }
        if changed {
            // The old bound still applies to the zero-artificial original
            // domain, but it cannot certify convergence of this cheaper LP.
            // Preserve only its selected proof channel as a tree floor.
            self.state.tree_bound_floor = self.state.tree_bound_floor.max(self.warm.bound);
            self.warm.bound = f64::NEG_INFINITY;
            self.warm.numerical_bound = f64::NEG_INFINITY;
            self.warm.certified_bound = f64::NEG_INFINITY;
            self.warm.stabilization = None;
            self.warm.clear_proximal();
        }
        Ok(())
    }

    /// Resume this node with a new time/iteration budget and no external registry.
    pub fn solve(&self, mut options: Options) -> Result<Output<D>> {
        let deadline = self.resume_deadline(&mut options)?;
        self.fork()?.solve_forked_at(options, deadline)
    }

    /// Consume an independent node fork already prepared by a tree worker or
    /// strong-branching trial. Its owner has performed the normal fork reset;
    /// copying the LP, graph and warm column pool again would be redundant.
    pub(crate) fn solve_forked(self, mut options: Options) -> Result<Output<D>> {
        let deadline = self.resume_deadline(&mut options)?;
        self.solve_forked_at(options, deadline)
    }

    fn resume_deadline(&self, options: &mut Options) -> Result<Instant> {
        validate_options(options)?;
        if !options.synchronized_services.is_empty() {
            if options.synchronized_services
                != self
                    .state
                    .synchronized_services
                    .iter()
                    .map(|s| s.definition.clone())
                    .collect::<Vec<_>>()
            {
                return Err(invalid(
                    "snapshot synchronization definitions cannot be changed",
                ));
            }
            options.synchronized_services.clear();
        }
        Instant::now()
            .checked_add(options.cg.time_limit)
            .ok_or_else(|| invalid("network time limit"))
    }

    fn solve_forked_at(mut self, mut options: Options, deadline: Instant) -> Result<Output<D>> {
        // A retained tree floor belongs to the snapshot's selected proof
        // channel. In particular, a numerical floor is not a certificate.
        if self.warm.bound_mode != options.cg.bound_mode {
            self.state.tree_bound_floor = f64::NEG_INFINITY;
        }
        for oracle in &mut self.state.oracles {
            oracle.solver.restore_node_state()?;
        }
        self.state.deadline = deadline;
        options.cg.warm_start = Some(self.warm.into_owned());
        run(self.master, self.state, options)
    }

    /// Append a shared additive row to both retained columns and future pricing.
    /// Omitted networks have zero coefficients. The caller supplies a valid cut
    /// or branch restriction; this method does not infer cut validity.
    pub fn with_row(&self, row: Row, coefficients: &[RowCoefficients]) -> Result<Self> {
        self.with_row_class(
            row,
            coefficients,
            cg::CurvatureRowClass::Ordinary,
            None,
            &[],
        )
    }

    /// Append a valid dynamic cut row, excluding it from automatic curvature
    /// distance measurements while retaining its ordinary pricing coefficients.
    pub fn with_dynamic_row(&self, row: Row, coefficients: &[RowCoefficients]) -> Result<Self> {
        self.with_row_class(row, coefficients, cg::CurvatureRowClass::Dynamic, None, &[])
    }

    /// Install a full-coordinate witness row with an independently established
    /// residual RHS. Activity descriptors must be present before coefficients
    /// and artificial columns are installed.
    pub(super) fn with_residual_activity_row(
        &self,
        row: Row,
        linear: &[RowCoefficients],
        activities: &[AccumulatedActivityCoefficients],
        residual_rhs: f64,
    ) -> Result<Self> {
        self.with_row_class(
            row,
            linear,
            cg::CurvatureRowClass::Ordinary,
            Some(residual_rhs),
            activities,
        )
    }

    fn with_row_class(
        &self,
        row: Row,
        coefficients: &[RowCoefficients],
        class: cg::CurvatureRowClass,
        residual_rhs: Option<f64>,
        activities: &[AccumulatedActivityCoefficients],
    ) -> Result<Self> {
        if !row.rhs.is_finite() || residual_rhs.is_some_and(|rhs| !rhs.is_finite()) {
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
        activity_rows::install(&mut node.state, shared, activities)?;
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
            appended_row_value(
                oracle,
                &route.route,
                node.state.shared_offset,
                shared,
                physical,
                coefficients.iter().find(|c| c.network == route.network),
            )
        };
        let rhs = if let Some(rhs) = residual_rhs {
            rhs
        } else {
            let mut rhs = row.rhs;
            for fixed in &node.state.fixed_routes {
                rhs -= fixed.quantity * route_value(&fixed.route)?;
            }
            rhs
        };
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
        let mut pending = crate::master::formulation::PendingRows::new(physical);
        pending.push(row.sense, rhs, column_entries);
        pending.install(&mut node.master)?;
        node.state.shared_rows.push(row);
        node.state.curvature_rows.push(cg::CurvatureRow {
            class,
            active: true,
        });
        retirement::sync_rhs(&mut node.state, &node.master);
        // Row dimensions and objectives differ; the old smoothing center is
        // not a dual point for this child. Keep valid inherited lower bounds.
        node.warm.stabilization = None;
        node.warm.clear_proximal();
        let penalty = node.state.ordinary_artificial_cost;
        for (_, column) in
            cg::append_artificials(&mut node.master, physical..physical + 1, penalty)?
        {
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
                || o.activity_rows.iter().any(|entry| entry.row == shared_row)
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
        let mut child = self.with_row(Row { sense, rhs }, &[coefficients])?;
        child.state.node_depth = self.state.node_depth.saturating_add(1);
        Ok(child)
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
        // The node is immutable during scoring. Replay a referenced route's
        // coefficients once, then reuse them for all eligible assignment rows.
        // Distinguish registry tags from fixed-route positions in the cache.
        let mut coefficient_cache = HashMap::new();
        for shared_row in 0..self.state.shared_rows.len() {
            for oracle in &self.state.oracles {
                if self
                    .validate_assignment(shared_row, oracle.network.id)
                    .is_err()
                {
                    continue;
                }
                let physical = self.state.shared_offset + shared_row;
                let mut coefficient = |key: (bool, usize), route: &NetworkRoute| -> Result<f64> {
                    if route.network != oracle.network.id {
                        return Ok(0.);
                    }
                    let coefficients = match coefficient_cache.entry(key) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(entries(oracle, &route.route, self.state.shared_offset)?)
                        }
                    };
                    Ok(coefficients
                        .binary_search_by_key(&physical, |&(row, _)| row)
                        .map_or(0., |i| coefficients[i].1))
                };
                let mut value = 0.;
                for (column, metadata) in self.warm.columns.iter().enumerate() {
                    if let Some(metadata) = metadata {
                        if metadata.tag < 0 {
                            continue;
                        }
                        value += self.master.values()[column]
                            * coefficient(
                                (false, metadata.tag as usize),
                                &self.state.routes[metadata.tag as usize],
                            )?;
                    }
                }
                for (index, fixed) in self.state.fixed_routes.iter().enumerate() {
                    value += fixed.quantity * coefficient((true, index), &fixed.route)?;
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
                    candidate.network == route.network
                        && candidate.route.arcs == route.route.arcs
                        && candidate.route.coupled_values == route.route.coupled_values
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
        let mut rhs = node.state.rhs.clone();
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
        node.state.rhs = rhs;
        node.master
            .set_rhs(&retirement::physical_rhs(&node.state))?;
        node.master.objective_offset = offset;
        node.state.fixed_routes.push(FixedRoute {
            route: route.as_ref().clone(),
            quantity,
        });
        node.warm.stabilization = None;
        node.warm.clear_proximal();
        Ok(node)
    }
}

// Use the graph's exact-real resource replay rather than trusting the rounded
// resource profile returned by labeling.
pub(super) fn integer_route_feasible<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<bool> {
    let problem = super::resource_groups::domain_graph(oracle)?;
    Ok(problem.exact_route_feasible(route).is_some()
        && oracle
            .solver
            .route_satisfies_relaxation(&route.nodes, &route.arcs)?)
}

/// Fixed reductions replay original contributions even when a later branch
/// contradicts the fixed schedule. That contradiction can prove infeasibility;
/// it must not be mistaken for an inexact arithmetic reduction.
pub(super) fn original_route_feasible<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<bool> {
    let problem = oracle.network.graph.as_ref();
    Ok(problem.exact_route_feasible(route).is_some()
        && oracle
            .solver
            .route_satisfies_relaxation(&route.nodes, &route.arcs)?)
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
    master.set_rhs(&retirement::physical_rhs(state))?;
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
        let coefficients = retirement::active_entries(state, oracle, &seed.route)?;
        let cost = original_costs::route_cost(oracle, &seed.route)?;
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
        Arc::make_mut(&mut state.tags).insert(key, tag);
        Arc::make_mut(&mut state.routes).push(Arc::new(route));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/curvature_lifecycle_tests.rs"]
mod curvature_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/artificial_lifecycle_tests.rs"]
mod artificial_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/appended_row_tests.rs"]
mod appended_row_tests;
