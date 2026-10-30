use super::*;

impl<const D: usize> LabelingEngine<D> {
    pub fn solve(&mut self, g: &Graph<D>, controls: &Controls) -> Result<PricingResult> {
        self.solve_internal(g, controls, false)
    }

    /// Retain completed routes and statistics on label/deadline exhaustion.
    /// An interrupted result is never exact; arithmetic/state errors propagate.
    pub fn solve_partial(&mut self, g: &Graph<D>, controls: &Controls) -> Result<PricingResult> {
        self.solve_internal(g, controls, true)
    }

    fn solve_internal(
        &mut self,
        g: &Graph<D>,
        controls: &Controls,
        allow_partial: bool,
    ) -> Result<PricingResult> {
        if g.coupled_kernel()
            .is_some_and(|kernel| kernel.has_negative_paired_objectives())
        {
            return Err(crate::incomplete(
                "negative paired objectives require source-control master pricing",
            ));
        }
        let effective = controls.for_graph(g)?;
        let controls = &effective;
        self.cuts.presence.validate_engine(g.arcs.len())?;
        self.resource_error.store(false, Ordering::Relaxed);
        self.coupled_completion = std::array::from_fn(|_| Default::default());
        self.ensure_reversal_costs(g)?;
        if controls.max_columns == 0 || controls.max_labels == 0 {
            return Err(invalid("positive pricing limits required"));
        }
        if let Some(direction) = self.pending_middle.take() {
            self.grid.move_middle(g, direction);
        }
        self.extension_checks.fill(0);
        self.filter_checks.fill(0);
        self.half_ready = false;
        self.full_ready = false;
        self.last_costs.clear();
        self.last_costs
            .extend(g.arcs.iter().map(|a| a.reduced_cost));
        self.forward
            .reset(self.grid.forward.buckets.iter().map(Vec::len));
        self.backward
            .reset(self.grid.backward.buckets.iter().map(Vec::len));
        let ordinary = !self.has_cuts() && self.branching.resources == 0;
        let branch_only = !self.has_cuts() && self.branching.resources != 0;
        self.forward.labels.set_resource_mode(
            ordinary,
            branch_only,
            self.coupled_present || self.cuts.presence.active(),
        );
        self.backward.labels.set_resource_mode(
            ordinary,
            branch_only,
            self.coupled_present || self.cuts.presence.active(),
        );
        self.statistics = Statistics::default();
        self.root(g, true)?;
        if self.runtime_bidirectional() && !self.consumption_joins {
            self.root(g, false)?;
        }
        let started = Instant::now();
        self.load_completion = self.prepare_load_completion(g, started, controls.time_limit);
        let completion_setup = started.elapsed().as_secs_f64();
        if controls.time_limit.is_zero() {
            return Err(incomplete("pricing hard time limit exceeded"));
        }
        let mut pool = SolutionPool::new(
            g.vertices.len(),
            controls.max_columns,
            self.controls.group_limit,
        );
        let mut concatenation_tests = 0;
        let outcome: Result<()> = (|| {
            let parallel = self.runtime_bidirectional() && !self.consumption_joins;
            if parallel {
                (self.statistics.forward_time, self.statistics.backward_time) =
                    self.run_both_directions(g, controls, started, &mut pool)?;
            } else {
                self.run_direction(g, true, controls, started, &mut pool)?;
                self.statistics.forward_time = started.elapsed().as_secs_f64();
            }
            self.refresh_coupled_completion(g);
            // The native solver balances the half-way point right after backward
            // extension, so the move survives a concatenation timeout.
            if parallel && self.phase.exact() {
                let f = self
                    .forward
                    .buckets
                    .iter()
                    .flatten()
                    .map(|b| b.active.len())
                    .sum::<usize>() as f64;
                let b = self
                    .backward
                    .buckets
                    .iter()
                    .flatten()
                    .map(|b| b.active.len())
                    .sum::<usize>() as f64;
                if f / b > 1.2 {
                    self.pending_middle = Some(false);
                } else if b / f > 1.2 {
                    self.pending_middle = Some(true);
                }
            }
            // Small heuristic sweeps may perform fewer than 64 extensions and
            // have no concatenation pass. They still consume the call deadline.
            if started.elapsed() >= controls.time_limit {
                return Err(incomplete("pricing hard time limit exceeded"));
            }
            if self.runtime_bidirectional() {
                if !parallel {
                    self.statistics.backward_time =
                        started.elapsed().as_secs_f64() - self.statistics.forward_time;
                }
                let concatenation_started = Instant::now();
                self.update_unreachable(self.consumption_joins);
                // Crossing labels share a traversal stack; each traversal resets
                // its contents while retaining capacity from the previous label.
                let mut work = Vec::new();
                for internal in 0..self.grid.order.len() {
                    let v = self.grid.order[internal];
                    let mut cutoff = pool.cutoff(v);
                    for &f in &self.forward.crossing[v] {
                        let label = self.forward.labels.view(f);
                        let mut coordinate = self.grid.coordinate(label.resources());
                        if self.consumption_joins {
                            for r in 0..self.grid.dimensions {
                                coordinate[r] = (self.grid.upper[internal][r]
                                    + self.grid.lower[internal][r])
                                    - coordinate[r];
                            }
                        }
                        let Some(bucket) =
                            self.grid
                                .bucket_id(self.consumption_joins, internal, coordinate)
                        else {
                            continue;
                        };
                        let key = label.key()
                            + self.cuts.knapsack.partial_cost(g, label.resources(), true);
                        let base_key = key;
                        let key = self.ctx().resource_completion_key(
                            g,
                            true,
                            label.coupled_state(),
                            label.presence_state(),
                            key,
                        );
                        work.clear();
                        work.push((bucket, 0usize));
                        while let Some((k, min)) = work.pop() {
                            let row = &self.storage(self.consumption_joins).buckets[internal][k];
                            if self.ctx().completion_sum(
                                key,
                                row.completion_bound(self.cuts.knapsack.active),
                            ) >= cutoff
                            {
                                continue;
                            }
                            if let Some(scan_key) = self.ctx().bucket_completion(
                                g,
                                true,
                                label,
                                base_key,
                                BucketId {
                                    vertex: internal,
                                    bucket: k,
                                },
                                row.hot.key.first().copied(),
                                cutoff,
                            ) {
                                for &b in &row.active {
                                    let other = self.storage(self.consumption_joins).labels.view(b);
                                    if self.ctx().completion_sum(scan_key, other.key()) >= cutoff {
                                        break;
                                    }
                                    concatenation_tests += 1;
                                    if let Some(cost) = self.ctx().join_cost(g, v, label, other) {
                                        if cost < cutoff
                                            && controls.max_route_arcs.is_none_or(|limit| {
                                                label
                                                    .hops()
                                                    .checked_add(other.hops())
                                                    .is_some_and(|hops| hops <= limit)
                                            })
                                        {
                                            pool.insert(v, cost, 0., (Some(f), Some(b)));
                                            cutoff = pool.cutoff(v);
                                        }
                                    }
                                }
                            }
                            for p in self.grid.direction(self.consumption_joins).buckets[internal]
                                [k]
                                .predecessors
                                .iter()
                                .rev()
                            {
                                if p.bucket.bucket >= min {
                                    work.push((p.bucket.bucket, min.max(p.minimum)));
                                }
                            }
                        }
                    }
                    if started.elapsed() > controls.time_limit {
                        return Err(incomplete("pricing concatenation time limit exceeded"));
                    }
                }
                self.statistics.concatenation_time = concatenation_started.elapsed().as_secs_f64();
                if self.phase.exact() && !self.consumption_joins {
                    self.half_ready = self.controls.direction_flags == 1;
                }
            }
            if self.phase.exact() && self.controls.filter_other_buckets {
                for v in 0..self.grid.order.len() {
                    if self.grid.forward.depth[v] > 0
                        && self.filter_checks[v] > self.extension_checks[v].wrapping_mul(3)
                    {
                        self.grid.forward.depth[v] =
                            (self.grid.forward.depth[v] as f64 * 0.9).floor() as i32;
                        if self.grid.bidirectional {
                            self.grid.backward.depth[v] =
                                (self.grid.backward.depth[v] as f64 * 0.9).floor() as i32;
                        }
                    }
                }
            }
            Ok(())
        })();
        let interrupted = match outcome {
            Ok(()) => false,
            Err(error)
                if allow_partial
                    && matches!(
                        error.message.as_str(),
                        "pricing label limit exceeded"
                            | "pricing hard time limit exceeded"
                            | "pricing concatenation time limit exceeded"
                    ) =>
            {
                self.half_ready = false;
                self.full_ready = false;
                true
            }
            Err(error) => return Err(error),
        };
        self.statistics.concatenation_tests = concatenation_tests;
        self.statistics.forward_labels = self
            .forward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.active.len())
            .sum();
        self.statistics.backward_labels = self
            .backward
            .buckets
            .iter()
            .flatten()
            .map(|b| b.active.len())
            .sum();
        self.statistics.elapsed = started.elapsed().as_secs_f64();
        // The minimum is the first candidate that can be recorded.
        let mut minimum = NO_ROUTE_COST;
        let mut recorded = false;
        let mut columns = vec![];
        for c in pool.into_sorted() {
            let Some(route) = self.reconstruct(g, c.value, c.cost)? else {
                continue;
            };
            if !recorded {
                minimum = c.cost;
                recorded = true;
            }
            if !controls.negative_only || c.cost < controls.zero_threshold - EPSILON {
                columns.push(route);
            }
        }
        if self.resource_error.load(Ordering::Relaxed) {
            return Err(incomplete("coupled resource arithmetic or state failure"));
        }
        if self.statistics.completion_pruned > 0 {
            minimum = minimum.min(self.controls.load_completion_threshold);
            // Pruned pricing labels cannot serve as full-domain fixing or
            // enumeration completion labels at a larger incumbent cutoff.
            self.half_ready = false;
            self.full_ready = false;
        }
        if std::env::var_os("BALDES_DEBUG_COMPLETION").is_some() && self.load_completion.is_some() {
            eprintln!(
                "load completion: setup={completion_setup:.6} pruned={} generated={} expanded={} subset={}",
                self.statistics.completion_pruned,
                self.statistics.generated,
                self.statistics.expanded,
                self.load_completion.as_ref().map_or(0, |b| b.subset_size())
            );
        }
        self.load_completion = None;
        Ok(PricingResult {
            columns,
            minimum_reduced_cost: minimum,
            exact: self.phase.exact() && !interrupted,
            statistics: self.statistics.clone(),
        })
    }
    pub(super) fn update_unreachable(&mut self, forward: bool) {
        for v in 0..self.grid.order.len() {
            for k in 0..self.grid.direction(forward).buckets[v].len() {
                let b = &self.grid.direction(forward).buckets[v][k];
                if b.reachable {
                    continue;
                }
                let lower = b
                    .predecessors
                    .iter()
                    .map(|p| self.storage(forward).buckets[v][p.bucket.bucket].lower_cost)
                    .fold(NO_ROUTE_COST, f64::min);
                let knapsack_lower = b
                    .predecessors
                    .iter()
                    .map(|p| self.storage(forward).buckets[v][p.bucket.bucket].knapsack_lower_cost)
                    .fold(NO_ROUTE_COST, f64::min);
                self.storage_mut(forward).buckets[v][k].lower_cost = lower;
                self.storage_mut(forward).buckets[v][k].knapsack_lower_cost = knapsack_lower;
            }
        }
    }

    pub fn continue_after_halfway(&mut self, g: &Graph<D>, controls: &Controls) -> Result<()> {
        if g.coupled_kernel()
            .is_some_and(|kernel| kernel.has_negative_paired_objectives())
        {
            return Err(crate::incomplete(
                "negative paired objectives require source-control master pricing",
            ));
        }
        if self.full_ready
            && self
                .last_costs
                .iter()
                .copied()
                .eq(g.arcs.iter().map(|a| a.reduced_cost))
        {
            return Ok(());
        }
        if !self.half_ready
            || !self
                .last_costs
                .iter()
                .copied()
                .eq(g.arcs.iter().map(|a| a.reduced_cost))
        {
            return Err(invalid("continuation requires fresh exact half-labels"));
        }
        self.half_ready = false;
        self.full_ready = false;
        self.coupled_completion = std::array::from_fn(|_| Default::default());
        let saved = self.controls.clone();
        let phase = self.phase;
        self.controls.direction_flags = 2;
        self.controls.sink_concatenation = false;
        self.phase.labels_per_bucket = i64::MAX;
        let result = (|| {
            for v in 0..self.forward.crossing.len() {
                // Insertion changes the label arena and buckets, not these
                // crossing IDs. End each ID borrow before mutating the engine.
                for position in 0..self.forward.crossing[v].len() {
                    let id = self.forward.crossing[v][position];
                    let mut label = self.forward.labels.view(id).to_owned();
                    self.vertex_step(&mut label);
                    let hot = HotLabel::of(&label);
                    self.insert(g, true, &label, hot, controls);
                }
            }
            let grid = &self.grid;
            self.forward
                .labels
                .unexpand_where(|resources| grid.coordinate(&resources)[0] > grid.middle + EPSILON);
            let started = Instant::now();
            let mut pool = SolutionPool::new(g.vertices.len(), 1, 0);
            self.run_direction(g, true, controls, started, &mut pool)?;
            self.backward.labels.unexpand_where(|_| true);
            self.run_direction(g, false, controls, started, &mut pool)?;
            self.update_unreachable(true);
            self.update_unreachable(false);
            if self.resource_error.load(Ordering::Relaxed) {
                return Err(incomplete(
                    "coupled resource continuation arithmetic failure",
                ));
            }
            self.refresh_coupled_completion(g);
            self.full_ready = true;
            Ok(())
        })();
        self.controls = saved;
        self.phase = phase;
        result
    }
}
