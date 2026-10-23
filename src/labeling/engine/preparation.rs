use super::*;

impl<const D: usize> LabelingEngine<D> {
    /// Carry the bucket graph (including fixed arcs), adaptive state and cuts.
    /// Labels depend on the node's duals and are rebuilt in the child.
    pub(crate) fn fork_for_node(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            forward: LabelStorage::new(self.forward.buckets.iter().map(Vec::len)),
            backward: LabelStorage::new(self.backward.buckets.iter().map(Vec::len)),
            statistics: Statistics::default(),
            controls: self.controls.clone(),
            phase: self.phase,
            ng: self.ng.clone(),
            branching: self.branching.clone(),
            forbidden_arcs: self.forbidden_arcs.clone(),
            half_ready: false,
            full_ready: false,
            last_costs: vec![],
            pending_middle: self.pending_middle,
            consumption_joins: self.consumption_joins,
            symmetric_costs: self.symmetric_costs.clone(),
            reverse: self.reverse.clone(),
            extension_checks: vec![0; self.extension_checks.len()],
            filter_checks: vec![0; self.filter_checks.len()],
            cuts: self.cuts.clone(),
            coupled_costs: self.coupled_costs,
            coupled_present: self.coupled_present,
            coupled_identity: self.coupled_identity.clone(),
            coupled_completion: std::array::from_fn(|_| Default::default()),
            resource_error: AtomicBool::new(false),
        }
    }

    pub fn prepare(g: &Graph<D>, controls: EngineControls<D>, phase: Phase) -> Result<Self> {
        phase.validate()?;
        let mut grid = BucketGraph::prepare(
            g,
            &controls.steps,
            controls.target_buckets,
            controls.bidirectional,
        )?;
        let reverse = crate::graph::symmetry::reverse_arcs(g).unwrap_or_default();
        let consumption_joins = controls.bidirectional
            && g.coupled_kernel()
                .is_none_or(|k| !g.resources.is_empty() && k.reversal_eligible(&reverse))
            && controls.use_symmetric_execution
            && !reverse.is_empty()
            && crate::graph::symmetry::consumption_symmetric(g, &grid, &reverse);
        if consumption_joins {
            grid.bidirectional = false;
            grid.middle = (grid.lower[0][0] + grid.upper[grid.order.len() - 1][0]) * 0.5;
            for v in 0..grid.forward.buckets.len() {
                for k in 0..grid.forward.buckets[v].len() {
                    let arcs = &grid.forward.buckets[v][k].arcs;
                    let flags: Vec<_> = arcs.iter().map(|a| grid.flags(g, true, a)).collect();
                    for (a, flags) in grid.forward.buckets[v][k].arcs.iter_mut().zip(flags) {
                        a.flags = flags;
                    }
                }
            }
        }
        let forward = LabelStorage::new(grid.forward.buckets.iter().map(Vec::len));
        let backward = LabelStorage::new(grid.backward.buckets.iter().map(Vec::len));
        Ok(Self {
            grid,
            forward,
            backward,
            statistics: Statistics::default(),
            controls,
            phase,
            ng: PreparedNg::new(g)?,
            branching: crate::pricing::restrictions::RyanFosterResources::prepare(g, &[])?,
            forbidden_arcs: vec![],
            half_ready: false,
            full_ready: false,
            last_costs: vec![],
            pending_middle: None,
            consumption_joins,
            symmetric_costs: if consumption_joins {
                crate::graph::symmetry::reversal_costs(g, &reverse).unwrap()
            } else {
                vec![]
            },
            reverse,
            extension_checks: vec![0; g.vertices.len()],
            filter_checks: vec![0; g.vertices.len()],
            cuts: EngineCuts::default(),
            coupled_costs: g.coupled_kernel().is_some_and(|k| k.has_costs()),
            coupled_present: g.has_coupled_resources(),
            coupled_identity: g.coupled_kernel().map(|k| k.cache_identity().clone()),
            coupled_completion: std::array::from_fn(|_| Default::default()),
            resource_error: AtomicBool::new(false),
        })
    }

    pub(super) fn storage(&self, forward: bool) -> &LabelStorage<D> {
        if forward {
            &self.forward
        } else {
            &self.backward
        }
    }

    pub(super) fn storage_mut(&mut self, forward: bool) -> &mut LabelStorage<D> {
        if forward {
            &mut self.forward
        } else {
            &mut self.backward
        }
    }

    pub(super) fn root(&mut self, g: &Graph<D>, forward: bool) -> Result<()> {
        let vertex = if forward { g.source } else { g.sink };
        let mut resources = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
        for (r, kind) in g.resources.iter().enumerate() {
            let w = g.vertices[vertex].bounds[r];
            resources[r] = g
                .root_resource(r, forward)
                .ok_or_else(|| invalid("root resource outside vertex windows"))?;
            if resources[r] < w.lower || resources[r] > w.upper {
                return Err(invalid("root resource outside vertex window"));
            }
            if *kind == ResourceKind::Equality && !forward && w.lower != w.upper {
                return Err(invalid(
                    "backward equality resource requires a fixed sink window",
                ));
            }
        }
        let internal = self.grid.internal[vertex];
        if g.has_time_windows() && !forward {
            for (axis, &slot) in self.grid.resource_slots.iter().enumerate() {
                resources[slot] = g
                    .project_resource(
                        vertex,
                        slot,
                        g.vertices[vertex].bounds[slot],
                        resources[slot].min(self.grid.upper[internal][axis]),
                        false,
                    )
                    .ok_or_else(|| invalid("backward root outside effective windows"))?;
            }
        }
        let bucket = self
            .grid
            .bucket_id(forward, internal, self.grid.coordinate(&resources))
            .ok_or_else(|| invalid("root outside bucket grid"))?;
        let label = StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: internal,
                bucket,
            },
            vertex,
            cost: 0.,
            original_cost: 0.,
            resources,
            ng: self.ng.vertex_bits[vertex],
            branch_state: None,
            presence_state: self.cuts.presence.empty_state(),
            coupled_state: g.coupled_kernel().map(|k| {
                if self.consumption_joins {
                    k.reversal_root(forward)
                } else {
                    k.root(forward)
                }
            }),
            parent: None,
            arc: None,
            hops: 0,
            expanded: false,
            cut_words: [0; LIMITED_WORDS],
            additional_cost: 0.,
            half_potential: self
                .cuts
                .info
                .vertex_cut_potential
                .get(vertex)
                .map_or(0.0, |&p| 0.5 * p),
        };
        let mut label = label;
        (label.cut_words, label.additional_cost) = self.root_cut_state(vertex);
        let storage = self.storage_mut(forward);
        let id = storage.retain(label);
        storage.root = Some(id);
        let hot = storage.hot[id];
        storage.buckets[internal][bucket].active.push(id);
        storage.buckets[internal][bucket].hot.push(hot);
        storage.buckets[internal][bucket].next_id = 1;
        Ok(())
    }
    pub fn costs_unchanged(&self, g: &Graph<D>) -> bool {
        self.last_costs
            .iter()
            .copied()
            .eq(g.arcs.iter().map(|a| a.reduced_cost))
    }
    pub(super) fn runtime_bidirectional(&self) -> bool {
        self.controls.bidirectional
            && (self.controls.bidirectional_mode == 1
                || (self.controls.bidirectional_mode == 2 && self.phase.exact()))
    }
    /// Refresh conservative cost potentials after dual changes. Costs with
    /// nonzero reversal cycles require ordinary directional labeling.
    pub(super) fn ensure_reversal_costs(&mut self, g: &Graph<D>) -> Result<()> {
        let same_resources = match (&self.coupled_identity, g.coupled_kernel()) {
            (None, None) => true,
            (Some(id), Some(kernel)) => std::sync::Arc::ptr_eq(id, kernel.cache_identity()),
            _ => false,
        };
        if same_resources && !self.consumption_joins {
            return Ok(());
        }
        if same_resources
            && !self.cuts.presence.active()
            && g.coupled_kernel()
                .is_none_or(|k| k.reversal_eligible(&self.reverse))
            && self.reverse.iter().enumerate().all(|(i, &r)| {
                self.forbidden_arcs.get(i).copied().unwrap_or(false)
                    == self.forbidden_arcs.get(r).copied().unwrap_or(false)
            })
        {
            if let Some(costs) = crate::graph::symmetry::reversal_costs(g, &self.reverse) {
                self.symmetric_costs = costs;
                return Ok(());
            }
        }
        let mut controls = self.controls.clone();
        controls.use_symmetric_execution = false;
        let mut rebuilt = Self::prepare(g, controls, self.phase)?;
        rebuilt.branching = self.branching.clone();
        rebuilt.forbidden_arcs = self.forbidden_arcs.clone();
        rebuilt.cuts.knapsack = self.cuts.knapsack.clone();
        rebuilt.cuts.presence = self.cuts.presence.clone();
        if !rebuilt.load_cuts(g, self.cuts.inputs.clone(), self.cuts.kpath.clone()) {
            return Err(invalid(
                "cut memory exceeds capacity after symmetry fallback",
            ));
        }
        *self = rebuilt;
        Ok(())
    }
}
