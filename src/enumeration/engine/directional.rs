use super::*;

impl<const D: usize> Enumeration<D> {
    /// Directional elementary enumeration with a shared path-equivalence store.
    /// Completion is supplied by the complete pricing stores at these same costs.
    pub fn run_directional<F: Fn(bool, &Label<D>) -> bool>(
        g: &Graph<D>,
        grid: &BucketGraph,
        controls: &Controls,
        completion: F,
    ) -> Result<Self> {
        Self::run_with_symmetry(g, grid, controls, &[], None, completion)
    }

    pub fn run_with_symmetry<F: Fn(bool, &Label<D>) -> bool>(
        g: &Graph<D>,
        grid: &BucketGraph,
        controls: &Controls,
        reverse: &[usize],
        cuts: Option<CutTables>,
        completion: F,
    ) -> Result<Self> {
        let consumption = !reverse.is_empty();
        if consumption && g.has_time_windows() {
            return Err(invalid(
                "multi-time-window enumeration requires ordinary directional buckets",
            ));
        }
        if consumption && cuts.and_then(|c| c.presence).is_some_and(|p| p.active()) {
            return Err(invalid(
                "presence enumeration requires ordinary directional buckets",
            ));
        }
        if consumption && !crate::graph::symmetry::reduced_costs_symmetric(g, reverse) {
            let costs = crate::graph::symmetry::reversal_costs(g, reverse).ok_or_else(|| {
                invalid("nonconservative reduced costs cannot use symmetric enumeration")
            })?;
            let mut transformed = g.clone();
            for (arc, cost) in transformed.arcs.iter_mut().zip(costs) {
                arc.reduced_cost = cost;
            }
            let mut result =
                Self::run_with_symmetry(&transformed, grid, controls, reverse, cuts, completion)?;
            result.pruning.symmetry_transformed = true;
            return Ok(result);
        }
        let depot_mask = crate::graph::symmetry::depot_mask(g);
        if (!consumption && grid.backward.buckets.is_empty()) || grid.resource_slots.is_empty() {
            return Self::run(g, grid, controls, cuts);
        }
        g.validate()?;
        if controls.label_limit == 0 || controls.path_limit == 0 || !controls.cutoff.is_finite() {
            return Err(invalid("invalid enumeration controls"));
        }
        if let Some(presence) = cuts.and_then(|c| c.presence) {
            presence.validate_engine(g.arcs.len())?;
        }
        let preserve = controls.preserve_arc_paths
            || cuts.and_then(|c| c.presence).is_some_and(|p| p.active());
        let started = Instant::now();
        let mut forward = Storage::new(controls.partitions)?;
        let mut backward = Storage::new(controls.partitions)?;
        forward.preserve_arc_paths = preserve;
        backward.preserve_arc_paths = preserve;
        let mut pruning = PruningFrontier::for_run(preserve);
        let mut paths = PathStorage::new(controls.partitions)?;
        paths.preserve_arc_paths = preserve;
        let mut crossing = vec![vec![]; g.vertices.len()];
        let mut roots = vec![];
        for direction in [true, false] {
            let vertex = if direction { g.source } else { g.sink };
            let mut root = Label {
                id: 0,
                vertex,
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
            };
            for r in 0..g.resources.len() {
                root.resources[r] = g
                    .root_resource(r, direction)
                    .ok_or_else(|| invalid("enumeration root outside vertex windows"))?;
            }
            if g.has_time_windows() && !direction {
                let internal = grid.internal[vertex];
                for (axis, &slot) in grid.resource_slots.iter().enumerate() {
                    root.resources[slot] = g
                        .project_resource(
                            vertex,
                            slot,
                            g.vertices[vertex].bounds[slot],
                            root.resources[slot].min(grid.upper[internal][axis]),
                            false,
                        )
                        .ok_or_else(|| invalid("enumeration root outside effective windows"))?;
                }
            }
            if !root.add_elements(&g.vertices[vertex].elements) {
                return Err(invalid("invalid enumeration root elements"));
            }
            if let Some(c) = &cuts {
                c.to_vertex(&mut root);
            }
            roots.push(root);
        }
        for direction in [true, false] {
            if consumption && !direction {
                continue;
            }
            let storage = if direction {
                &mut forward
            } else {
                &mut backward
            };
            let root = &roots[if direction { 0 } else { 1 }];
            let opposite_root = &roots[if direction && !consumption { 1 } else { 0 }];
            storage.insert(g, root.clone(), direction);
            let step = controls.label_limit / 50;
            let mut checkpoint = 0;
            while let Some(from) = storage.pop() {
                if started.elapsed() > controls.time_limit {
                    return Err(incomplete("enumeration time limit exceeded"));
                }
                let generated = storage.arena.len();
                if generated > controls.label_limit {
                    return Err(incomplete("enumeration label budget exceeded"));
                }
                if generated - checkpoint >= step {
                    let ratio = generated as f64 / controls.label_limit as f64;
                    if 1. - ratio / 1.5 < storage.pending as f64 / generated as f64 {
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
                let Some(bucket) = grid.bucket_id(direction, v, grid.coordinate(&from.resources))
                else {
                    continue;
                };
                for arc in &grid.direction(direction).buckets[v][bucket].arcs {
                    let physical = &g.arcs[arc.physical];
                    let mut next = from.clone();
                    next.parent = Some(from.id);
                    next.arc = Some(arc.physical);
                    next.vertex = if direction {
                        physical.head
                    } else {
                        physical.tail
                    };
                    next.hops += 1;
                    next.reduced_cost = physical.reduced_cost + from.reduced_cost;
                    next.original_cost = from.original_cost + physical.cost;
                    if !next.add_elements(&physical.elements) {
                        continue;
                    }
                    let mut q = next.resources;
                    if let Some(jump) = arc.jump {
                        for (r, &slot) in grid.resource_slots.iter().enumerate() {
                            if g.resources[slot] == ResourceKind::Equality {
                                continue;
                            }
                            let value = grid.direction(direction).corners[v][jump][r];
                            q[slot] = if direction {
                                q[slot].max(value)
                            } else {
                                q[slot].min(value)
                            };
                        }
                    }
                    let Some(q) = extend_resources(g, physical, &q, direction) else {
                        continue;
                    };
                    next.resources = q.clone();
                    if let Some(c) = &cuts {
                        c.along_arc(&mut next, direction, arc.physical)?;
                    }
                    let terminal = next.vertex == if direction { g.sink } else { g.source };
                    if !terminal {
                        // Do not project an already non-elementary prefix into
                        // completion. Keep the candidate vertex out of the NG
                        // projection until the callback has accepted it.
                        if !next.can_add_elements(&g.vertices[next.vertex].elements) {
                            continue;
                        }
                        let before = grid.coordinate(&q)[0] <= grid.middle + EPSILON;
                        if !direction && before {
                            continue;
                        }
                        if !completion(direction, &next) {
                            pruning.completion_rejections += 1;
                            continue;
                        }
                        if direction && !before {
                            crossing[next.vertex].push(next);
                            continue;
                        }
                    }
                    if terminal {
                        // Reaching the source window is insufficient for an
                        // equality resource: the suffix must reach the actual
                        // initial value, not another value in that window.
                        if !direction
                            && g.resources.iter().enumerate().any(|(r, kind)| {
                                *kind == ResourceKind::Equality
                                    && next.resources[r] != roots[0].resources[r]
                            })
                        {
                            continue;
                        }
                        if let Some(c) = &cuts {
                            // The opposite root owns the terminal's cut state.
                            next.reduced_cost = if direction {
                                c.join(
                                    g,
                                    consumption,
                                    &next,
                                    opposite_root,
                                    ((next.reduced_cost + 0.0) + 0.0) + 0.0,
                                )?
                            } else {
                                c.join(
                                    g,
                                    consumption,
                                    opposite_root,
                                    &next,
                                    ((0.0 + next.reduced_cost) + 0.0) + 0.0,
                                )?
                            };
                        }
                        // Symmetric cost joins use the forward source seed,
                        // but elementarity still checks the physical terminal.
                        // Comparing with that source seed would reject every
                        // forward terminal because the path already owns it.
                        let terminal_root = &roots[if direction { 1 } else { 0 }];
                        if next
                            .visited
                            .iter()
                            .zip(&terminal_root.visited)
                            .any(|(a, b)| a & b != 0)
                        {
                            continue;
                        }
                        if next.reduced_cost > controls.cutoff - EPSILON {
                            pruning.terminal(next.reduced_cost);
                            continue;
                        }
                        if !next.add_elements(&g.vertices[terminal_root.vertex].elements) {
                            continue;
                        }
                        let mut arcs = vec![arc.physical];
                        let mut parent = Some(from.id);
                        while let Some(id) = parent {
                            let label = &storage.arena[id];
                            if let Some(a) = label.arc {
                                arcs.push(a);
                            }
                            parent = label.parent;
                        }
                        if direction {
                            arcs.reverse();
                        }
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
                        if !next.add_elements(&g.vertices[next.vertex].elements) {
                            continue;
                        }
                        if let Some(c) = &cuts {
                            c.to_vertex(&mut next);
                        }
                        storage.insert(g, next, direction);
                    }
                }
            }
        }
        // Bucket, partition and original-cost order determine which equivalent
        // elementary route survives. Preserve that order across crossing joins.
        let opposite = if consumption { &forward } else { &backward };
        let mut buckets: Vec<Vec<Vec<usize>>> = grid
            .direction(consumption)
            .buckets
            .iter()
            .map(|row| vec![vec![]; row.len()])
            .collect();
        for row in &opposite.partitions {
            for &id in row {
                let label = &opposite.arena[id];
                let vertex = grid.internal[label.vertex];
                if let Some(k) =
                    grid.bucket_id(consumption, vertex, grid.coordinate(&label.resources))
                {
                    buckets[vertex][k].push(id);
                }
            }
        }
        let label_key = |l: &Label<D>| {
            l.key_with(
                cuts.and_then(|c| c.info.vertex_cut_potential.get(l.vertex))
                    .copied()
                    .unwrap_or(0.0),
            )
        };
        // A full-memory join can reduce the sum of the label keys. Include
        // every possible negative field correction before pruning the sorted
        // opposite labels; otherwise improving cut contributions lose routes.
        let minimum_cut_join = cuts.map_or(0., |c| {
            c.full
                .fields
                .iter()
                .map(|f| (if f.covering { -f.cost } else { f.cost }).min(0.))
                .sum::<f64>()
        });
        for row in &mut buckets {
            for bucket in row {
                bucket.sort_by(|&a, &b| {
                    label_key(&opposite.arena[a])
                        .total_cmp(&label_key(&opposite.arena[b]))
                        .then(a.cmp(&b))
                });
            }
        }
        for vertex in 0..grid.order.len() {
            let v = grid.order[vertex];
            for f in &crossing[v] {
                let mut q = grid.coordinate(&f.resources);
                if consumption {
                    for r in 0..grid.dimensions {
                        q[r] = grid.upper[vertex][r] + grid.lower[vertex][r] - q[r];
                    }
                }
                let Some(k) = grid.bucket_id(consumption, vertex, q) else {
                    continue;
                };
                let mut pending = vec![(k, 0usize)];
                while let Some((bucket, min)) = pending.pop() {
                    let key = label_key(f);
                    for &id in &buckets[vertex][bucket] {
                        let b = &opposite.arena[id];
                        let ordered_bound = if let Some(presence) =
                            cuts.and_then(|c| c.presence).filter(|p| p.active())
                        {
                            presence
                                .join_correction_lower_bound(&f.presence_state)?
                                .and_then(|bound| {
                                    crate::core::rounding::add_down(key, label_key(b))
                                        .and_then(|n| {
                                            crate::core::rounding::add_down(n, minimum_cut_join)
                                        })
                                        .and_then(|n| crate::core::rounding::add_down(n, bound))
                                })
                                .unwrap_or(f64::NEG_INFINITY)
                        } else {
                            key + label_key(b) + minimum_cut_join
                        };
                        if ordered_bound > controls.cutoff - EPSILON {
                            pruning.ordered(ordered_bound);
                            break;
                        }
                        let cost = match &cuts {
                            Some(c) => c.join(
                                g,
                                consumption,
                                f,
                                b,
                                ((f.reduced_cost + b.reduced_cost) + 0.0) + 0.0,
                            )?,
                            None => f.reduced_cost + b.reduced_cost,
                        };
                        if cost > controls.cutoff - EPSILON {
                            pruning.joined(cost);
                            continue;
                        }
                        if controls
                            .max_route_arcs
                            .is_some_and(|limit| f.hops + b.hops > limit)
                            || f.visited.iter().zip(&b.visited).zip(&depot_mask).any(
                                |((x, y), depot)| {
                                    x & y & if consumption { !depot } else { u64::MAX } != 0
                                },
                            )
                        {
                            continue;
                        }
                        if !g.resources.iter().enumerate().all(|(r, kind)| match kind {
                            ResourceKind::Disposable => {
                                if consumption {
                                    f.resources[r] + b.resources[r]
                                        - g.initial[r]
                                            .unwrap_or(g.vertices[g.source].bounds[r].lower)
                                        <= g.vertices[g.sink].bounds[r].upper + EPSILON
                                } else {
                                    f.resources[r] <= b.resources[r] + EPSILON
                                }
                            }
                            ResourceKind::Equality => f.resources[r] == b.resources[r],
                        }) {
                            continue;
                        }
                        let mut arcs = vec![];
                        let mut label = f;
                        loop {
                            if let Some(a) = label.arc {
                                arcs.push(a);
                            }
                            let Some(parent) = label.parent else {
                                break;
                            };
                            label = &forward.arena[parent];
                        }
                        arcs.reverse();
                        let mut suffix = vec![];
                        let mut label = b;
                        loop {
                            if let Some(a) = label.arc {
                                suffix.push(a);
                            }
                            let Some(parent) = label.parent else {
                                break;
                            };
                            label = &opposite.arena[parent];
                        }
                        if consumption {
                            arcs.extend(suffix.into_iter().map(|a| reverse[a]));
                        } else {
                            arcs.extend(suffix);
                        }
                        let mut visited = f.visited;
                        for (x, y) in visited.iter_mut().zip(&b.visited) {
                            *x |= *y;
                        }
                        let mut element_hash = f.element_hash.wrapping_add(b.element_hash);
                        if consumption {
                            for &e in &g.vertices[g.source].elements {
                                let e = e as u32;
                                element_hash =
                                    element_hash.wrapping_sub(e.wrapping_mul(e).wrapping_mul(e));
                            }
                            for &e in &g.vertices[g.sink].elements {
                                visited[e / 64] |= 1 << (e % 64);
                                let e = e as u32;
                                element_hash =
                                    element_hash.wrapping_add(e.wrapping_mul(e).wrapping_mul(e));
                            }
                        }
                        paths.insert(Path {
                            reduced_cost: cost,
                            original_cost: f.original_cost + b.original_cost,
                            visited,
                            element_hash,
                            arcs,
                        });
                        if paths.len > controls.path_limit {
                            return Err(incomplete("enumeration path budget exceeded"));
                        }
                    }
                    for p in grid.direction(consumption).buckets[vertex][bucket]
                        .predecessors
                        .iter()
                        .rev()
                    {
                        if p.bucket.bucket >= min {
                            pending.push((p.bucket.bucket, min.max(p.minimum)));
                        }
                    }
                }
                if started.elapsed() > controls.time_limit {
                    return Err(incomplete("enumeration concatenation time limit exceeded"));
                }
            }
        }
        let generated = forward.arena.len() + backward.arena.len();
        Ok(Self {
            labels: forward,
            paths,
            completed: true,
            generated,
            pruning,
        })
    }
}
