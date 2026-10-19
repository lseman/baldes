use super::*;

impl LabelingEngine {
    pub fn has_cuts(&self) -> bool {
        !self.cuts.inputs.is_empty() || !self.cuts.kpath.is_empty()
    }

    /// Limited-memory cut tables for both directions. False when a vertex
    /// would need more than twenty words. Labels are discarded.
    pub fn load_cuts(
        &mut self,
        g: &Graph,
        inputs: Vec<LimitedInput>,
        kpath: Vec<LimitedKPathInput>,
    ) -> bool {
        self.half_ready = false;
        self.full_ready = false;
        let mut cuts = EngineCuts {
            inputs,
            kpath,
            knapsack: self.cuts.knapsack.clone(),
            ..Default::default()
        };
        if !cuts.inputs.is_empty() || !cuts.kpath.is_empty() {
            if !state::prepare_vertices(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
            ) {
                return false;
            }
            // The recorded preparation contract omits negative vertex
            // increments. Covering rows use those increments, so install
            // them for actual pricing and retain their owners for dual updates.
            for (row, input) in cuts.inputs.iter().enumerate() {
                for (vertex, &increment) in input.vertex_coefficient.iter().enumerate() {
                    if increment >= 0 {
                        continue;
                    }
                    let Some(slot) = cuts.limited.dominance_cut[vertex]
                        .iter()
                        .position(|&owner| owner == row)
                    else {
                        continue;
                    };
                    let field = cuts.info.vertices[vertex].dominance[slot];
                    cuts.info.vertices[vertex]
                        .support
                        .push(state::VertexSupport {
                            mask: field.mask,
                            denominator: field.denominator,
                            word: field.word,
                            shift: field.shift,
                            increment,
                            covering: input.covering,
                            cost: input.dual,
                        });
                    cuts.limited.support_cut[vertex].push(row);
                }
            }
            state::prepare_arcs(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
                true,
            );
            state::prepare_arcs(
                &mut cuts.info,
                &mut cuts.limited,
                g,
                &cuts.inputs,
                &cuts.kpath,
                false,
            );
        }
        let words = cuts
            .limited
            .positions
            .iter()
            .flatten()
            .copied()
            .filter(|&p| p >= 0)
            .max()
            .map_or(0, |position| position as usize / 32 + 1);
        self.forward.labels.set_word_limit(words);
        self.backward.labels.set_word_limit(words);
        self.cuts = cuts;
        true
    }

    /// update_lim_mem_duals: same cut rows, new duals. Labels are discarded.
    pub fn update_cut_duals(&mut self, inputs: Vec<LimitedInput>, kpath: Vec<LimitedKPathInput>) {
        self.half_ready = false;
        self.full_ready = false;
        state::update_duals(
            &mut self.cuts.info,
            &self.cuts.limited,
            &inputs,
            &kpath,
            true,
            true,
        );
        self.cuts.inputs = inputs;
        self.cuts.kpath = kpath;
        self.cuts.full = None;
    }

    /// updateComplBoundsWithResConsKnapsackCutDuals: preserve the raw
    /// dominance bounds and rebuild the resource-cost completion bounds.
    pub(super) fn update_knapsack_bounds(&mut self, g: &Graph, forward: bool) {
        if !self.cuts.knapsack.active {
            return;
        }
        let components = self.grid.direction(forward).components.clone();
        for component in components {
            for id in component {
                let bucket = &self.storage(forward).buckets[id.vertex][id.bucket];
                let mut lower = NO_ROUTE_COST;
                for &index in &bucket.active {
                    let label = self.storage(forward).labels.view(index);
                    let cost = label.key()
                        + self
                            .cuts
                            .knapsack
                            .partial_cost(g, label.resources(), forward);
                    lower = lower.min(cost);
                }
                for predecessor in
                    &self.grid.direction(forward).buckets[id.vertex][id.bucket].predecessors
                {
                    lower = lower.min(
                        self.storage(forward).buckets[id.vertex][predecessor.bucket.bucket]
                            .knapsack_lower_cost,
                    );
                }
                self.storage_mut(forward).buckets[id.vertex][id.bucket].knapsack_lower_cost = lower;
            }
        }
    }

    /// Completion passes rebuild both directional lower-cost tables. Ordinary
    /// pricing only needs the opposite table for joins and must not be used as
    /// a substitute: an unreachable forward cell can still inherit a valid
    /// completion bound from its dominance predecessors.
    pub fn prepare_full_labels(&mut self, g: &Graph, controls: &Controls) -> Result<()> {
        self.ensure_reversal_costs(g)?;
        if self.costs_unchanged(g) {
            if self.full_ready {
                return Ok(());
            }
            if self.half_ready {
                return self.continue_after_halfway(g, controls);
            }
        }
        if !self.consumption_joins {
            self.grid.ensure_backward(g)?;
        }
        let saved = self.controls.clone();
        let phase = self.phase;
        self.controls.bidirectional = !self.consumption_joins;
        self.controls.bidirectional_mode = 1;
        self.controls.direction_flags = 3;
        self.controls.sink_concatenation = false;
        self.controls.dominance_mode = 0;
        self.phase = Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        };
        self.half_ready = false;
        self.full_ready = false;
        self.last_costs = g.arcs.iter().map(|a| a.reduced_cost).collect();
        self.forward
            .reset(self.grid.forward.buckets.iter().map(Vec::len));
        self.backward
            .reset(self.grid.backward.buckets.iter().map(Vec::len));
        let ordinary = !self.has_cuts() && self.branching.resources == 0;
        let branch_only = !self.has_cuts() && self.branching.resources != 0;
        self.forward.labels.set_mode(ordinary, branch_only);
        self.backward.labels.set_mode(ordinary, branch_only);
        self.statistics = Statistics::default();
        self.extension_checks.fill(0);
        self.filter_checks.fill(0);
        // A fixing completion is not a new pricing call. In particular, retain
        // the pending border movement for the next actual pricing call.
        let result = (|| {
            self.root(g, true)?;
            if !self.consumption_joins {
                self.root(g, false)?;
            }
            let started = Instant::now();
            let mut pool = SolutionPool::new(g.vertices.len(), 1, 0);
            self.run_direction(g, true, controls, started, &mut pool)?;
            if !self.consumption_joins {
                self.run_direction(g, false, controls, started, &mut pool)?;
            }
            self.update_unreachable(true);
            if !self.consumption_joins {
                self.update_unreachable(false);
            }
            self.full_ready = true;
            Ok(())
        })();
        self.controls = saved;
        self.phase = phase;
        result
    }
}
