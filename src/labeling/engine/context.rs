use super::*;
/// The read-only engine state a labeling sweep reads; both directions can
/// share it across threads.
#[derive(Clone, Copy)]
pub(super) struct Context<'a, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) grid: &'a BucketGraph,
    pub(super) ng: &'a PreparedNg,
    pub(super) branching: &'a crate::pricing::restrictions::RyanFosterResources,
    pub(super) forbidden_arcs: &'a [bool],
    pub(super) cuts: &'a EngineCuts,
    pub(super) controls: &'a EngineControls<D>,
    pub(super) phase: Phase,
    pub(super) consumption_joins: bool,
    pub(super) symmetric_costs: &'a [f64],
}
impl<'a, const D: usize> Context<'a, D> {
    pub(super) fn has_cuts(&self) -> bool {
        !self.cuts.inputs.is_empty() || !self.cuts.kpath.is_empty()
    }
    pub(super) fn runtime_bidirectional(&self) -> bool {
        self.controls.bidirectional
            && (self.controls.bidirectional_mode == 1
                || (self.controls.bidirectional_mode == 2 && self.phase.exact()))
    }
    /// The meeting vertex's NG bits and cut supports.
    pub(super) fn vertex_step(&self, label: &mut StoredLabel<D>) {
        self.vertex_step_with_branching::<true>(label);
    }
    #[inline(always)]
    pub(super) fn vertex_step_with_branching<const BRANCHING: bool>(
        &self,
        label: &mut StoredLabel<D>,
    ) {
        label.ng |= self.ng.vertex_bits[label.vertex];
        if BRANCHING {
            let valid = self.branching.visit(&mut label.branch_state, label.vertex);
            debug_assert!(valid, "extension checked the target's branch resource");
        }
        if let Some(v) = self.cuts.info.vertices.get(label.vertex) {
            state::apply_vertex_support(
                &mut label.cut_words,
                &mut label.additional_cost,
                &v.support,
            );
        }
    }
    /// Cut state of a root label at its terminal vertex.
    pub(super) fn root_cut_state(&self, vertex: usize) -> ([u32; LIMITED_WORDS], f64) {
        let mut words = [0; LIMITED_WORDS];
        let mut additional = 0.;
        if let Some(v) = self.cuts.info.vertices.get(vertex) {
            state::apply_vertex_support(&mut words, &mut additional, &v.support);
        }
        (words, additional)
    }
    /// Join of two labels at `vertex`: resources, then the
    /// cost (forward + backward reduced cost) + 0, then NG, then the cut join.
    /// The cost is 0 when resources reject and has no cut part when NG does.
    #[inline(always)]
    pub(super) fn concatenate(
        &self,
        g: &Graph<D>,
        vertex: usize,
        f: LabelRef<'_, D>,
        b: LabelRef<'_, D>,
    ) -> (bool, f64) {
        match (f, b) {
            (LabelRef::Ordinary(f, _), LabelRef::Ordinary(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Extended(f, _), LabelRef::Extended(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Branched(f, _), LabelRef::Branched(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Compact2(f, _), LabelRef::Compact2(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Compact4(f, _), LabelRef::Compact4(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Compact8(f, _), LabelRef::Compact8(b, _)) => {
                self.concatenate_records(g, vertex, f, b)
            }
            (LabelRef::Full(f), LabelRef::Full(b)) => self.concatenate_records(g, vertex, f, b),
            _ => self.concatenate_records(g, vertex, &f, &b),
        }
    }
    #[inline(always)]
    pub(super) fn concatenate_records<F: LabelData, B: LabelData>(
        &self,
        g: &Graph<D>,
        vertex: usize,
        f: &F,
        b: &B,
    ) -> (bool, f64) {
        if !self.branching.joins(f.branch_state(), b.branch_state())
            || !self.resources_join(g, f, b)
        {
            return (false, 0.);
        }
        let cost = (f.cost() + b.cost())
            + self
                .cuts
                .knapsack
                .join_cost(g, f.resources(), b.resources(), self.consumption_joins);
        let mut overlap = f.ng() & b.ng();
        if self.consumption_joins {
            for (bit, e) in self.ng.local_elements[vertex].iter().enumerate() {
                if g.vertices[g.source].elements.contains(e)
                    || g.vertices[g.sink].elements.contains(e)
                {
                    overlap &= !(1 << bit);
                }
            }
        }
        if overlap != 0 {
            return (false, cost);
        }
        let cost = if self.has_cuts() {
            state::concatenation_cost(
                &self.cuts.info,
                vertex,
                (f.cut_words(), f.additional_cost()),
                (b.cut_words(), b.additional_cost()),
                cost,
            )
        } else if self.cuts.presence.active() {
            cost + f.additional_cost() + b.additional_cost()
        } else {
            cost
        };
        if self.cuts.presence.active() {
            let Ok(correction) = self
                .cuts
                .presence
                .join_cost(f.presence_state(), b.presence_state())
            else {
                return (false, f64::INFINITY);
            };
            return (true, cost + correction);
        }
        (true, cost)
    }
    pub fn join_cost(
        &self,
        g: &Graph<D>,
        vertex: usize,
        f: LabelRef<'_, D>,
        b: LabelRef<'_, D>,
    ) -> Option<f64> {
        match self.concatenate(g, vertex, f, b) {
            (true, cost) => Some(cost),
            _ => None,
        }
    }
    /// Join of a label reaching its end vertex with the opposite root (forward:
    /// the backward root, or the forward one for symmetric joins; backward: the
    /// forward root at the source). The join's result is not tested.
    pub(super) fn terminal_cost(&self, g: &Graph<D>, forward: bool, label: &StoredLabel<D>) -> f64 {
        if !self.has_cuts() && !self.cuts.knapsack.active && !self.cuts.presence.active() {
            return label.cost;
        }
        let root_forward = !forward || self.consumption_joins;
        let root_vertex = if root_forward { g.source } else { g.sink };
        let mut root = label.clone();
        root.vertex = root_vertex;
        root.cost = 0.;
        root.ng = self.ng.vertex_bits[root_vertex];
        root.branch_state = None;
        root.presence_state = self.cuts.presence.empty_state();
        for (r, _) in g.resources.iter().enumerate() {
            let w = g.vertices[root_vertex].bounds[r];
            root.resources[r] = if root_forward {
                g.initial[r].unwrap_or(w.lower)
            } else {
                w.upper
            };
        }
        (root.cut_words, root.additional_cost) = self.root_cut_state(root_vertex);
        if forward {
            self.concatenate(g, g.sink, label.into(), (&root).into()).1
        } else {
            self.concatenate(g, g.source, (&root).into(), label.into())
                .1
        }
    }
    #[inline(always)]
    pub(super) fn extend(
        &self,
        g: &Graph<D>,
        forward: bool,
        arc: &BucketArc,
        from: LabelRef<'_, D>,
    ) -> Option<StoredLabel<D>> {
        self.extend_resources_kept::<true>(g, forward, arc, from)
            .map(|(label, _)| label)
    }
    /// `extend`, also returning the label's resources as a value: reading
    /// them back from the new label (two 8-byte stores, one 16-byte load)
    /// stalls on store forwarding.
    #[inline(always)]
    pub(super) fn extend_resources_kept<const BRANCHING: bool>(
        &self,
        g: &Graph<D>,
        forward: bool,
        arc: &BucketArc,
        from: LabelRef<'_, D>,
    ) -> Option<(StoredLabel<D>, crate::graph::ResourceValues<D>)> {
        match from {
            LabelRef::Ordinary(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Extended(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Branched(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Compact2(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Compact4(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Compact8(label, id) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, id)
            }
            LabelRef::Full(label) => {
                self.extend_record::<BRANCHING, _>(g, forward, arc, label, label.id)
            }
        }
    }
    #[inline(always)]
    pub(super) fn extend_record<const BRANCHING: bool, L: LabelData>(
        &self,
        g: &Graph<D>,
        forward: bool,
        arc: &BucketArc,
        from: &L,
        id: usize,
    ) -> Option<(StoredLabel<D>, crate::graph::ResourceValues<D>)> {
        if self.has_cuts() {
            self.extend_record_with_cuts::<BRANCHING, true, L>(g, forward, arc, from, id)
        } else {
            self.extend_record_with_cuts::<BRANCHING, false, L>(g, forward, arc, from, id)
        }
    }
    #[inline(always)]
    pub(super) fn extend_record_with_cuts<const BRANCHING: bool, const CUTS: bool, L: LabelData>(
        &self,
        g: &Graph<D>,
        forward: bool,
        arc: &BucketArc,
        from: &L,
        id: usize,
    ) -> Option<(StoredLabel<D>, crate::graph::ResourceValues<D>)> {
        let physical = &g.arcs[arc.physical];
        if self
            .forbidden_arcs
            .get(arc.physical)
            .copied()
            .unwrap_or(false)
            || physical.reduced_cost >= NO_ROUTE_COST
        {
            return None;
        }
        let mut q = crate::graph::ResourceValues::<D>::from_slice(from.resources());
        if self.phase.jump_arcs {
            if let Some(jump) = arc.jump {
                let point = self.grid.direction(forward).corners[arc.from.vertex][jump];
                for (axis, &slot) in self.grid.resource_slots.iter().enumerate() {
                    if g.resources[slot] == ResourceKind::Equality {
                        continue;
                    }
                    q[slot] = if forward {
                        q[slot].max(point[axis])
                    } else {
                        q[slot].min(point[axis])
                    };
                }
            }
        }
        let resources: crate::graph::ResourceValues<D> =
            crate::labeling::ScalarResourceKernel::<D>::extend_active(g, physical, q.0, forward)?
                .into();
        let vertex = if forward {
            physical.head
        } else {
            physical.tail
        };
        if BRANCHING && !self.branching.can_visit(from.branch_state(), vertex) {
            return None;
        }
        let transition = if forward {
            &self.ng.forward[arc.physical]
        } else {
            &self.ng.backward[arc.physical]
        };
        let ng = transition.extend(from.ng(), 0)?;
        let internal = self.grid.internal[vertex];
        let bucket = self
            .grid
            .bucket_id(forward, internal, self.grid.coordinate(&resources))?;
        let mut additional_cost = from.additional_cost();
        let cut_words = if CUTS {
            let table = if forward {
                &self.cuts.info.forward
            } else {
                &self.cuts.info.backward
            };
            state::changed_along_arc(
                from.cut_words(),
                &mut additional_cost,
                &table[arc.physical].ordinary,
            )
        } else {
            let mut words = [0; LIMITED_WORDS];
            words[..from.cut_words().len()].copy_from_slice(from.cut_words());
            words
        };
        let presence_state = if self.cuts.presence.active() {
            let (next, increment) = self
                .cuts
                .presence
                .extend(from.presence_state(), arc.physical)
                .expect("validated engine presence state and cost range");
            additional_cost += increment;
            next
        } else {
            from.presence_state().clone()
        };
        Some((
            StoredLabel {
                id: 0,
                local_id: 0,
                bucket: BucketId {
                    vertex: internal,
                    bucket,
                },
                vertex,
                cost: self
                    .symmetric_costs
                    .get(arc.physical)
                    .copied()
                    .unwrap_or(physical.reduced_cost)
                    + from.cost(),
                original_cost: physical.cost + from.original_cost(),
                resources: resources.clone(),
                ng,
                // Like NG state, a crossing label excludes its meeting vertex.
                branch_state: if BRANCHING {
                    from.branch_state().clone()
                } else {
                    None
                },
                presence_state,
                parent: Some(id),
                arc: Some(arc.physical),
                hops: from.hops() + 1,
                expanded: false,
                cut_words,
                additional_cost,
                half_potential: self
                    .cuts
                    .info
                    .vertex_cut_potential
                    .get(vertex)
                    .map_or(0.0, |&p| 0.5 * p),
            },
            resources,
        ))
    }
    pub(super) fn resources_join<F: LabelData, B: LabelData>(
        &self,
        g: &Graph<D>,
        f: &F,
        b: &B,
    ) -> bool {
        g.resources.iter().enumerate().all(|(r, k)| match k {
            ResourceKind::Disposable => {
                if self.consumption_joins {
                    f.resources()[r] + b.resources()[r]
                        - g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower)
                        <= g.vertices[g.sink].bounds[r].upper + EPSILON
                } else {
                    f.resources()[r] <= b.resources()[r] + EPSILON
                }
            }
            // Consumption-mode concatenation checks only disposable
            // resources; equality resources are checked in value mode.
            ResourceKind::Equality => {
                self.consumption_joins || f.resources()[r] == b.resources()[r]
            }
        })
    }
}
