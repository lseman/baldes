// Branch-and-bound pricing over physical paths and source controls.
use super::*;
pub(super) fn branch_and_bound_price<const D: usize>(
    state: &State<D>,
    index: usize,
    duals: &[f64],
    work: &mut usize,
) -> Option<Result<(Vec<Route>, f64)>> {
    let oracle = state.oracles.get(index)?;
    let original_costs = state
        .original_variable_costs
        .iter()
        .find(|model| model.network == oracle.network.id);
    let physical = physical_suffix_bounds(
        oracle,
        state.shared_offset,
        duals,
        original_costs,
        work,
        state.deadline,
    )?;
    let graph = &oracle.network.graph;
    let hops = physical.suffix.len().checked_sub(1)?;
    // The scalar suffix remains a walk relaxation, but this proof traversal
    // only needs route-domain paths. Avoid revisiting vertices when physical
    // simplicity is declared or implied by hard groups; bounded walks that
    // permit repeats continue to explore them.
    let vertex_simple_domain = graph.route_domain() == crate::graph::RouteDomain::PhysicalSimple
        || graph.bounded_walks_are_vertex_simple();
    let prune_repeated_vertices = vertex_simple_domain && !graph.is_acyclic();
    let Some(global_tail) = physical
        .suffix
        .get(hops)?
        .get(graph.source)
        .copied()
        .flatten()
    else {
        return Some(Ok((Vec::new(), 0.)));
    };
    let Some(global_constant) =
        crate::core::rounding::add_down(physical.route_constant, physical.other_coupled_lower)
    else {
        return None;
    };
    // This suffix relaxation bounds every admissible walk. If branch
    // refinement later hits its budget, retained pricing can still use this
    // certified minimum bound alongside the source-validated columns found.
    let global_lower_bound = crate::core::rounding::add_down(global_tail, global_constant)?;
    // The universal bound closes pricing immediately when it is nonnegative;
    // candidate generation cannot produce a negative column in that case.
    if global_lower_bound >= 0. {
        return Some(Ok((Vec::new(), global_lower_bound)));
    }
    let mut outgoing = vec![Vec::new(); graph.vertices.len()];
    for (arc, data) in graph.arcs.iter().enumerate() {
        if arc % 256 == 0 && Instant::now() >= state.deadline {
            return Some(Ok((Vec::new(), global_lower_bound)));
        }
        let Some(left) = work.checked_sub(1) else {
            return Some(Ok((Vec::new(), global_lower_bound)));
        };
        *work = left;
        // The suffix rows represent walks of at most the indexed hop count.
        // If an arc's head cannot reach the sink with the largest possible
        // remaining budget, no DFS frame can use this arc to complete a
        // route. Preserve the original arc order for every retained edge.
        let reachable = hops
            .checked_sub(1)
            .and_then(|remaining| physical.suffix.get(remaining))
            .and_then(|row| row.get(data.head))
            .copied()
            .flatten()
            .is_some();
        if !reachable {
            continue;
        }
        outgoing.get_mut(data.tail)?.push(arc);
    }
    if physical.exact_additive {
        for arcs in &mut outgoing {
            arcs.sort_by(|&left, &right| {
                let bound = |arc: usize| {
                    physical
                        .suffix
                        .get(hops.saturating_sub(1))
                        .and_then(|row| row.get(graph.arcs[arc].head))
                        .copied()
                        .flatten()
                        .map(|tail| physical.arc_costs[arc] + tail)
                        .unwrap_or(f64::INFINITY)
                };
                bound(left).total_cmp(&bound(right))
            });
        }
    }
    let mut initial_resources =
        crate::graph::ResourceValues::<D>::filled(graph.resources.len(), 0.);
    for (resource, value) in initial_resources
        .iter_mut()
        .enumerate()
        .take(graph.resources.len())
    {
        let Some(left) = work.checked_sub(1) else {
            return Some(Ok((Vec::new(), global_lower_bound)));
        };
        *work = left;
        let Some(root) = graph.root_resource(resource, true) else {
            return Some(Ok((Vec::new(), 0.)));
        };
        *value = root;
    }
    struct Frame<const D: usize> {
        vertex: usize,
        remaining: usize,
        prefix: f64,
        resources: crate::graph::ResourceValues<D>,
        paired_base_domains: Option<Vec<PairedPrefixDomain>>,
        path_len: usize,
        next_arc: usize,
        entered: bool,
    }
    let mut stack = vec![Frame::<D> {
        vertex: graph.source,
        remaining: hops,
        prefix: 0.,
        resources: initial_resources,
        paired_base_domains: paired_prefix_roots(graph, work, state.deadline),
        path_len: 0,
        next_arc: 0,
        entered: false,
    }];
    let mut path = Vec::new();
    let mut visited_vertices = vec![false; graph.vertices.len()];
    if prune_repeated_vertices {
        visited_vertices[graph.source] = true;
    }
    let pop_frame =
        |stack: &mut Vec<Frame<D>>, path: &mut Vec<usize>, visited_vertices: &mut [bool]| {
            if let Some(frame) = stack.pop() {
                if prune_repeated_vertices && frame.vertex != graph.source {
                    visited_vertices[frame.vertex] = false;
                }
                if let Some(parent) = stack.last() {
                    path.truncate(parent.path_len);
                }
            }
        };
    let mut iterations = 100_000;
    let mut minimum = f64::INFINITY;
    let mut routes = Vec::new();
    let mut incumbent_upper = f64::INFINITY;
    let mut seen = HashSet::new();
    // Inputs other than the exact starting ceiling and hop count are fixed
    // per charge for this pricing call. Cache only successful refinements.
    let mut ceiling_cache = HashMap::new();
    // Full-domain raw slope hulls are immutable during this pricing call and
    // can be reused across cyclic prefixes of the same physical arc.
    let mut paired_arc_hulls = HashMap::new();
    // Exact paired prefix states can recur on cyclic bounded walks. Cache
    // only completed extensions, never arithmetic or budget uncertainty.
    type PrefixEntry = (usize, u64, u64, u64, u64);
    type PrefixKey = (usize, usize, [PrefixEntry; 8]);
    let mut paired_extensions: HashMap<PrefixKey, Option<Vec<PairedPrefixDomain>>> = HashMap::new();
    // A bounded walk may revisit an arc with exactly the same scalar state.
    // The graph is immutable for this pricing call, so scalar projection is
    // deterministic for a physical arc and all D input binary64 bit patterns.
    // Keep inactive slots in the key: scalar extension preserves them.
    let cache_scalar_projections = !vertex_simple_domain && !graph.resources.is_empty() && D <= 32;
    let mut scalar_projections: HashMap<
        (usize, [u64; D]),
        Option<crate::graph::ResourceValues<D>>,
    > = HashMap::new();
    // Repeated cyclic prefixes can also share the complete dependent-value
    // enclosure at a given depth. All other refinement inputs are immutable
    // during this pricing call. Keep the memo bounded even on large graphs.
    let cache_refinements = physical
        .paired_terminal_charges
        .iter()
        .map(|charge| charge.terminals.len())
        .try_fold(0usize, usize::checked_add)
        .is_some_and(|count| count >= 4);
    let mut refinement_cache: HashMap<(usize, Vec<(usize, u64, u64)>), f64> = HashMap::new();
    while !stack.is_empty() {
        if Instant::now() >= state.deadline {
            return Some(Ok((routes, global_lower_bound)));
        }
        let top = stack.len() - 1;
        if !stack[top].entered {
            let Some(left) = work.checked_sub(1) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            *work = left;
            let frame = &stack[top];
            let Some(tail) = physical
                .suffix
                .get(frame.remaining)
                .and_then(|row| row.get(frame.vertex))
                .copied()
                .flatten()
            else {
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            };
            let Some(prefix_tail) = crate::core::rounding::add_down(frame.prefix, tail) else {
                return Some(Ok((routes, global_lower_bound)));
            };
            // This is the same directed sum already checked for the global
            // bound; its operands are immutable throughout this traversal.
            let Some(mut bound) = crate::core::rounding::add_down(prefix_tail, global_constant)
            else {
                return Some(Ok((routes, global_lower_bound)));
            };
            // The unrefined physical relaxation already bounds every route
            // below this prefix. A nonnegative value is sufficient to prune;
            // calculating paired refinements can only tighten that bound and
            // is unnecessary for the negative-column search. Retain this
            // valid, potentially looser bound for the reported minimum.
            if bound >= 0. || (physical.exact_additive && bound >= incumbent_upper) {
                minimum = minimum.min(bound);
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            }
            if let Some(domains) = frame.paired_base_domains.as_deref() {
                let key = (cache_refinements && domains.len() <= 32).then(|| {
                    (
                        frame.remaining,
                        domains
                            .iter()
                            .map(|domain| {
                                (
                                    domain.resource,
                                    domain.dependent_lower.to_bits(),
                                    domain.dependent_upper.to_bits(),
                                )
                            })
                            .collect(),
                    )
                });
                let cached = key
                    .as_ref()
                    .and_then(|key| refinement_cache.get(key).copied());
                let correction = if let Some(cached) = cached {
                    if Instant::now() >= state.deadline {
                        return Some(Ok((routes, global_lower_bound)));
                    }
                    cached
                } else {
                    let before = *work;
                    let Some(correction) = paired_prefix_terminal_correction(
                        graph,
                        &physical.paired_terminal_charges,
                        domains,
                        oracle,
                        state.shared_offset,
                        duals,
                        frame.remaining,
                        &mut ceiling_cache,
                        work,
                        state.deadline,
                    ) else {
                        return Some(Ok((routes, global_lower_bound)));
                    };
                    if let Some(key) = key {
                        if before - *work >= 4 && refinement_cache.len() < 1024 {
                            refinement_cache.insert(key, correction);
                        }
                    }
                    correction
                };
                let Some(refined) = crate::core::rounding::add_down(bound, correction) else {
                    return Some(Ok((routes, global_lower_bound)));
                };
                bound = refined;
            }
            if bound >= 0. || (physical.exact_additive && bound >= incumbent_upper) {
                minimum = minimum.min(bound);
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            }
            if frame.vertex == graph.sink {
                // Every DFS frame is reached by one ordered sequence of
                // physical arc indices; each outgoing arc is visited once
                // per frame. Consequently this exact sequence cannot be
                // reached twice, even when bounded walks revisit vertices.
                // Route/control identities still use `seen` in process_path.
                let first_route = routes.len();
                if process_path(
                    state,
                    oracle,
                    duals,
                    &path,
                    state.deadline,
                    work,
                    &mut iterations,
                    &mut minimum,
                    &mut routes,
                    &mut seen,
                )
                .is_err()
                {
                    return Some(Ok((routes, global_lower_bound)));
                }
                if physical.exact_additive {
                    for route in &routes[first_route..] {
                        let allowance = *work / 8;
                        let mut remaining = allowance;
                        let upper = additive::route_upper(
                            oracle,
                            state.shared_offset,
                            duals,
                            &path,
                            route,
                            &mut remaining,
                            state.deadline,
                        );
                        *work -= allowance - remaining;
                        if let Some(upper) = upper {
                            incumbent_upper = incumbent_upper.min(upper);
                        }
                    }
                }
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            }
            if frame.remaining == 0 {
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            }
            stack[top].entered = true;
        }
        let (arc, prefix, resources, path_len, remaining) = {
            let frame = &mut stack[top];
            let Some(&arc) = outgoing
                .get(frame.vertex)
                .and_then(|arcs| arcs.get(frame.next_arc))
            else {
                pop_frame(&mut stack, &mut path, &mut visited_vertices);
                continue;
            };
            frame.next_arc += 1;
            (
                arc,
                frame.prefix,
                frame.resources,
                frame.path_len,
                frame.remaining,
            )
        };
        let Some(left) = work.checked_sub(1) else {
            return Some(Ok((routes, global_lower_bound)));
        };
        *work = left;
        let head = graph.arcs[arc].head;
        if head == graph.source {
            continue;
        }
        if prune_repeated_vertices && visited_vertices[head] {
            continue;
        }
        // The walk relaxation includes every permitted physical continuation.
        // If it cannot reach the sink in the remaining hops, this child cannot
        // complete a route either. Check before projecting resources or
        // constructing paired-prefix domains for a dead cyclic branch.
        let Some(child_tail) = physical
            .suffix
            .get(remaining - 1)
            .and_then(|row| row.get(head))
        else {
            return Some(Ok((routes, global_lower_bound)));
        };
        let Some(child_tail) = *child_tail else {
            continue;
        };
        // Apply exactly the same directed additions used on entry to the
        // child frame. The physical suffix bounds every admissible child
        // continuation even before scalar and paired feasibility checks, so
        // a nonnegative result already proves it cannot produce a negative
        // column. Preserve it in the reported minimum, as the child would.
        let Some(child_prefix) = crate::core::rounding::add_down(prefix, physical.arc_costs[arc])
        else {
            return Some(Ok((routes, global_lower_bound)));
        };
        let Some(child_bound) = crate::core::rounding::add_down(child_prefix, child_tail)
            .and_then(|value| crate::core::rounding::add_down(value, global_constant))
        else {
            return Some(Ok((routes, global_lower_bound)));
        };
        if child_bound >= 0. || (physical.exact_additive && child_bound >= incumbent_upper) {
            minimum = minimum.min(child_bound);
            continue;
        }
        let Some(left) = work.checked_sub(graph.resources.len()) else {
            return Some(Ok((routes, global_lower_bound)));
        };
        *work = left;
        // Charge the same scalar work on cache hits, preserving the search's
        // work-limit accounting while avoiding repeated projection arithmetic.
        let key = cache_scalar_projections.then(|| (arc, resources.0.map(f64::to_bits)));
        let next_resources = match key.and_then(|key| scalar_projections.get(&key).copied()) {
            Some(cached) => {
                if Instant::now() >= state.deadline {
                    return Some(Ok((routes, global_lower_bound)));
                }
                cached
            }
            None => {
                let Some(projected) = extend_bounded_walk_resources(graph, arc, resources) else {
                    return Some(Ok((routes, global_lower_bound)));
                };
                if let Some(key) = key {
                    if scalar_projections.len() < 4096 {
                        scalar_projections.insert(key, projected);
                    }
                }
                projected
            }
        };
        let Some(next_resources) = next_resources else {
            continue;
        };
        // The parent frame remains on the DFS stack while this child is
        // prepared. Borrow its domains only after cheap arc/resource checks;
        // extension constructs the child's own domains and does not mutate
        // the parent. This avoids cloning every paired domain on each arc,
        // including arcs rejected by vertex or scalar-resource checks.
        let paired_base_domains = match stack[top].paired_base_domains.as_deref() {
            Some(current) => {
                let key = (!vertex_simple_domain && !current.is_empty() && current.len() <= 8)
                    .then(|| {
                        let mut entries = [(0, 0, 0, 0, 0); 8];
                        for (entry, domain) in entries.iter_mut().zip(current) {
                            *entry = (
                                domain.resource,
                                domain.base.lo.to_bits(),
                                domain.base.hi.to_bits(),
                                domain.dependent_lower.to_bits(),
                                domain.dependent_upper.to_bits(),
                            );
                        }
                        (arc, current.len(), entries)
                    });
                let cached = key.as_ref().and_then(|key| paired_extensions.get(key));
                let extension = if let Some(cached) = cached {
                    let Some(left) = work.checked_sub(current.len()) else {
                        return Some(Ok((routes, global_lower_bound)));
                    };
                    *work = left;
                    if Instant::now() >= state.deadline {
                        return Some(Ok((routes, global_lower_bound)));
                    }
                    Some(cached.clone())
                } else {
                    let before = *work;
                    let result = extend_paired_prefix_domains(
                        graph,
                        arc,
                        current,
                        &mut paired_arc_hulls,
                        work,
                        state.deadline,
                    );
                    if before - *work >= 4 && paired_extensions.len() < 1024 {
                        if let (Some(key), Some(completed)) = (key, &result) {
                            paired_extensions.insert(key, completed.clone());
                        }
                    }
                    result
                };
                match extension {
                    Some(Some(next)) => Some(next),
                    Some(None) => continue,
                    // Numerical uncertainty is not infeasibility. Disable only
                    // paired-base pruning below this prefix.
                    None => None,
                }
            }
            None => None,
        };
        path.truncate(path_len);
        path.push(arc);
        if prune_repeated_vertices {
            visited_vertices[head] = true;
        }
        stack.push(Frame {
            vertex: graph.arcs[arc].head,
            remaining: remaining - 1,
            prefix: child_prefix,
            resources: next_resources,
            paired_base_domains,
            path_len: path.len(),
            next_arc: 0,
            entered: false,
        });
    }
    Some(Ok((routes, if minimum.is_finite() { minimum } else { 0. })))
}

pub(in crate::master::model) fn price<const D: usize>(
    state: &State<D>,
    index: usize,
    duals: &[f64],
) -> Result<(Vec<Route>, f64)> {
    let oracle = &state.oracles[index];
    let mut work = 4_000_000;
    let mut iterations = 100_000;
    if let Some(result) = branch_and_bound_price(state, index, duals, &mut work) {
        return result;
    }
    let mut minimum = f64::INFINITY;
    let mut routes = vec![];
    let mut seen = HashSet::new();
    let mut visitor_error = None;
    let visited =
        bounds::columns::visit_physical_paths(oracle, &mut work, state.deadline, |arcs, work| {
            let result = process_path(
                state,
                oracle,
                duals,
                &arcs,
                state.deadline,
                work,
                &mut iterations,
                &mut minimum,
                &mut routes,
                &mut seen,
            );
            match result {
                Ok(()) => Ok(()),
                Err(error) => {
                    visitor_error = Some(error);
                    Err(crate::enumeration::PathDomainFailure::Arithmetic)
                }
            }
        });
    if let Some(error) = visitor_error {
        return Err(error);
    }
    visited.ok_or_else(|| {
        incomplete("paired source-control pricing path/control coverage incomplete")
    })?;
    Ok((
        routes,
        if minimum == f64::INFINITY {
            0.
        } else {
            minimum
        },
    ))
}
