use super::*;

impl<const D: usize> Enumeration<D> {
    /// Abort without publishing a partial universe. Existing successful path
    /// storage remains reusable for inspection at subsequent master duals.
    pub fn run(
        g: &Graph<D>,
        grid: &BucketGraph,
        controls: &Controls,
        cuts: Option<CutTables>,
    ) -> Result<Self> {
        g.validate()?;
        if controls.label_limit == 0 || controls.path_limit == 0 || !controls.cutoff.is_finite() {
            return Err(invalid("invalid enumeration controls"));
        }
        let preserve = controls.preserve_arc_paths
            || g.has_coupled_resources()
            || cuts.and_then(|c| c.presence).is_some_and(|p| p.active());
        let mut labels = Storage::new(controls.partitions)?;
        labels.preserve_arc_paths = preserve;
        let mut pruning = PruningFrontier::for_run(preserve);
        let mut paths = PathStorage::new(controls.partitions)?;
        paths.preserve_arc_paths = preserve;
        if let Some(presence) = cuts.and_then(|c| c.presence) {
            presence.validate_engine(g.arcs.len())?;
        }
        let started = Instant::now();
        let mut root = Label {
            id: 0,
            vertex: g.source,
            resources: crate::graph::ResourceValues::filled(g.resources.len(), 0.),
            reduced_cost: 0.,
            original_cost: 0.,
            visited: [0; 16],
            element_hash: 0,
            parent: None,
            arc: None,
            hops: 0,
            cut_words: [0; FULL_WORDS],
            cut_cost: 0.,
            presence_state: cuts.and_then(|c| c.presence).and_then(|p| p.empty_state()),
            coupled_state: g.coupled_kernel().map(|k| k.root(true)),
        };
        for r in 0..g.resources.len() {
            root.resources[r] = g
                .root_resource(r, true)
                .ok_or_else(|| invalid("initial resource outside source windows"))?;
        }
        if !root.add_elements(&g.vertices[g.source].elements) {
            return Err(invalid("invalid source elementarity sets"));
        }
        if let Some(c) = &cuts {
            c.to_vertex(&mut root);
        }
        labels.insert(g, root, true);
        let step = controls.label_limit / 50;
        let mut checkpoint = 0;
        let mut coupled_workspace = crate::graph::coupled::CoupledWorkspace::default();
        while let Some(from) = labels.pop() {
            if started.elapsed() > controls.time_limit {
                return Err(incomplete("enumeration time limit exceeded"));
            }
            let generated = labels.arena.len();
            if generated > controls.label_limit {
                return Err(incomplete("enumeration label budget exceeded"));
            }
            if generated - checkpoint >= step {
                let ratio = generated as f64 / controls.label_limit as f64;
                if 1. - ratio / 1.5 < labels.pending as f64 / generated as f64 {
                    return Err(incomplete("enumeration pending-label budget exceeded"));
                }
                checkpoint += step;
            }
            if controls
                .max_route_arcs
                .is_some_and(|limit| from.hops >= limit)
            {
                continue;
            }
            let v = grid.internal[from.vertex];
            let Some(bucket) = grid.bucket_id(true, v, grid.coordinate(&from.resources)) else {
                continue;
            };
            for arc in &grid.forward.buckets[v][bucket].arcs {
                let physical = &g.arcs[arc.physical];
                let mut next = from.clone_without_coupled();
                next.parent = Some(from.id);
                next.arc = Some(arc.physical);
                next.vertex = physical.head;
                next.hops += 1;
                if let Some(kernel) = g.coupled_kernel() {
                    let state = from
                        .coupled_state
                        .as_deref()
                        .ok_or_else(|| invalid("missing coupled enumeration state"))?;
                    let Some((mut next_state, _)) = kernel.extend_with_workspace(
                        state,
                        arc.physical,
                        true,
                        &mut coupled_workspace,
                    )?
                    else {
                        continue;
                    };
                    (next.reduced_cost, next.original_cost) = next_state.account_extension(
                        state,
                        physical.reduced_cost,
                        physical.cost,
                    )?;
                    next.coupled_state = Some(next_state);
                    if !next.reduced_cost.is_finite() || !next.original_cost.is_finite() {
                        return Err(incomplete("coupled enumeration cost overflow"));
                    }
                } else {
                    next.reduced_cost = physical.reduced_cost + from.reduced_cost;
                    next.original_cost = from.original_cost + physical.cost;
                }
                if !next.add_elements(&physical.elements) {
                    continue;
                }
                let mut q = next.resources;
                if let Some(jump) = arc.jump {
                    for (r, &slot) in grid.resource_slots.iter().enumerate() {
                        if g.resources[slot] == ResourceKind::Equality {
                            continue;
                        }
                        q[slot] = q[slot].max(grid.forward.corners[v][jump][r]);
                    }
                }
                let Some(q) = extend_resources(g, physical, &q, true) else {
                    continue;
                };
                next.resources = q;
                if let Some(c) = &cuts {
                    c.along_arc(&mut next, true, arc.physical)?;
                }
                if !next.add_elements(&g.vertices[next.vertex].elements) {
                    continue;
                }
                if let Some(c) = &cuts {
                    c.to_vertex(&mut next);
                }
                if next.vertex == g.sink {
                    if let Some(c) = &cuts {
                        let mut empty = next.clone();
                        empty.cut_words = [0; FULL_WORDS];
                        empty.cut_cost = 0.;
                        empty.presence_state = c.presence.and_then(|p| p.empty_state());
                        for r in 0..g.resources.len() {
                            empty.resources[r] = g.vertices[g.sink].bounds[r].upper;
                        }
                        next.reduced_cost =
                            c.join(g, false, &next, &empty, (next.reduced_cost + 0.0) + 0.0)?;
                    }
                    if next.reduced_cost > controls.cutoff - EPSILON {
                        pruning.terminal(next.reduced_cost);
                        continue;
                    }
                    let mut arcs = vec![arc.physical];
                    let mut parent = Some(from.id);
                    while let Some(id) = parent {
                        let label = &labels.arena[id];
                        if let Some(a) = label.arc {
                            arcs.push(a);
                        }
                        parent = label.parent;
                    }
                    arcs.reverse();
                    paths.insert(Path {
                        reduced_cost: next.reduced_cost,
                        original_cost: next.original_cost,
                        visited: next.visited,
                        element_hash: next.element_hash,
                        arcs,
                    });
                    if paths.len > controls.path_limit {
                        return Err(incomplete("enumeration path budget exceeded"));
                    }
                } else {
                    labels.insert(g, next, true);
                }
            }
        }
        let generated = labels.arena.len();
        Ok(Self {
            labels,
            paths,
            completed: true,
            generated,
            pruning,
        })
    }
}
