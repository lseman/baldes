//! Complete physical-simple paths under deterministic forward scalar resources.
//! No reduced-cost cutoff, bucket fixing, NG, cut or branch restriction is used.
use crate::graph::Graph;
use crate::labeling::extend_resources;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct PathDomainLimits {
    pub max_paths: usize,
    /// Logical stored arc entries; profiles add one fixed-size state per arc
    /// plus one per path. Allocator/graph-validation overhead is not included.
    pub max_stored_arcs: usize,
    pub max_work: usize,
    /// Absolute shared deadline; None deliberately disables the time limit.
    pub deadline: Option<Instant>,
}
impl Default for PathDomainLimits {
    fn default() -> Self {
        Self {
            max_paths: 100_000,
            max_stored_arcs: 1_000_000,
            max_work: 10_000_000,
            deadline: Instant::now().checked_add(Duration::from_secs(1)),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathDomainFailure {
    InvalidGraph,
    PathLimit,
    StoredArcLimit,
    WorkLimit,
    TimeLimit,
    Allocation,
    UnsupportedResources,
    Arithmetic,
}
impl std::fmt::Display for PathDomainFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "physical path collection: {self:?}")
    }
}
impl std::error::Error for PathDomainFailure {}

#[derive(Debug)]
pub struct CanonicalPath {
    arcs: Vec<usize>,
    resources: Vec<[f64; 2]>,
}
impl CanonicalPath {
    /// Positions in the borrowed original graph, retaining parallel arc identity.
    pub fn arc_positions(&self) -> &[usize] {
        &self.arcs
    }
    /// Source state and state after each arc; unused resource slots are zero.
    pub fn resource_profile(&self) -> &[[f64; 2]] {
        &self.resources
    }
}

/// A physical path with downward resource bounds, rather than a feasible
/// floating-point schedule. It must not be submitted as an integer witness.
#[derive(Debug)]
pub struct RelaxedResourcePath {
    arcs: Vec<usize>,
    lower_resources: Vec<[f64; 2]>,
}
impl RelaxedResourcePath {
    pub fn arc_positions(&self) -> &[usize] {
        &self.arcs
    }
    /// Source value and a lower bound after each arc on every exact-real
    /// feasible schedule along this path; unused slots are zero.
    pub fn resource_lower_bounds(&self) -> &[[f64; 2]] {
        &self.lower_resources
    }
}

/// A conservative superset of exact-real feasible, physically vertex-simple
/// source/sink paths for disposable scalar resources. Directed addition keeps
/// lower resource bounds; a path is rejected only when such a bound exceeds an
/// upper window. Extra infeasible paths may remain. These bounds are not route
/// schedules and do not establish backward/tolerant native-domain equivalence.
/// Cost cutoffs, fixing, NG, element sets, cuts and branches are ignored.
#[derive(Debug)]
pub struct CompleteResourceRelaxation<'g> {
    data: CollectedPaths<'g, RelaxedResourcePath>,
}
impl<'g> CompleteResourceRelaxation<'g> {
    /// The same path/work/storage/deadline controls as canonical collection.
    /// Unsupported kinds or arithmetic failure return no partial universe.
    pub fn collect(graph: &'g Graph, limits: PathDomainLimits) -> Result<Self, PathDomainFailure> {
        if limits.deadline.is_some_and(|end| Instant::now() >= end) {
            return Err(PathDomainFailure::TimeLimit);
        }
        if graph.resources.len() > 2 {
            return Err(PathDomainFailure::InvalidGraph);
        }
        if graph
            .resources
            .iter()
            .any(|r| *r != crate::graph::ResourceKind::Disposable)
        {
            return Err(PathDomainFailure::UnsupportedResources);
        }
        let data = collect_paths(
            graph,
            limits,
            extend_resource_lower_bounds,
            |arcs, lower_resources| RelaxedResourcePath {
                arcs,
                lower_resources,
            },
        )?;
        Ok(Self { data })
    }
    pub fn graph(&self) -> &'g Graph {
        self.data.graph
    }
    pub fn paths(&self) -> &[RelaxedResourcePath] {
        &self.data.paths
    }
    pub fn work_used(&self) -> usize {
        self.data.work_used
    }
    pub fn stored_arcs(&self) -> usize {
        self.data.stored_arcs
    }
}
fn extend_resource_lower_bounds(
    g: &Graph,
    arc: &crate::graph::Arc,
    mut q: [f64; 2],
) -> Result<Option<[f64; 2]>, PathDomainFailure> {
    for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
        let mut bounds = g.vertices[arc.head].bounds[r];
        if let Some(window) = arc.bounds[r] {
            bounds.lower = bounds.lower.max(window.lower);
            bounds.upper = bounds.upper.min(window.upper);
        }
        *value = crate::master::rounding::add_down(*value, arc.consumption[r])
            .ok_or(PathDomainFailure::Arithmetic)?
            .max(bounds.lower);
        if *value > bounds.upper {
            return Ok(None);
        }
    }
    Ok(Some(q))
}

/// Every vertex-simple source/sink path accepted by canonical forward resource
/// extension, including a direct source/sink arc. One earliest profile is kept.
/// This is not a proof that all feasible resource schedules or original routing
/// columns are represented. Such equivalence must be established by the caller.
/// The graph borrow prevents stale membership after graph mutation.
///
/// ```compile_fail
/// use baldes::{Graph, enumeration::{CompleteCanonicalPaths, PathDomainLimits}};
/// let mut graph = Graph::new(2, 0).unwrap();
/// let domain = CompleteCanonicalPaths::collect(&graph, PathDomainLimits::default()).unwrap();
/// graph.arcs.clear();
/// assert!(domain.paths().is_empty());
/// ```
#[derive(Debug)]
pub struct CompleteCanonicalPaths<'g> {
    data: CollectedPaths<'g, CanonicalPath>,
}
#[derive(Debug)]
struct CollectedPaths<'g, P> {
    graph: &'g Graph,
    paths: Vec<P>,
    work_used: usize,
    stored_arcs: usize,
}
struct Budget {
    remaining: usize,
    deadline: Option<Instant>,
    until_clock_check: usize,
}
impl Budget {
    fn spend(&mut self, work: usize) -> Result<(), PathDomainFailure> {
        // Collection is cooperatively timed. Poll once per small batch of
        // work, and always at explicit validation/completion checkpoints.
        if work == 0 || work >= self.until_clock_check {
            if self.deadline.is_some_and(|end| Instant::now() >= end) {
                return Err(PathDomainFailure::TimeLimit);
            }
            self.until_clock_check = 64;
        } else {
            self.until_clock_check -= work;
        }
        self.remaining = self
            .remaining
            .checked_sub(work)
            .ok_or(PathDomainFailure::WorkLimit)?;
        Ok(())
    }
}
struct Frame {
    vertex: usize,
    next_arc: usize,
    resources: [f64; 2],
}
fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), PathDomainFailure> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| PathDomainFailure::Allocation)
}
fn collect_paths<'g, P>(
    graph: &'g Graph,
    limits: PathDomainLimits,
    mut extend: impl FnMut(
        &Graph,
        &crate::graph::Arc,
        [f64; 2],
    ) -> Result<Option<[f64; 2]>, PathDomainFailure>,
    mut make_path: impl FnMut(Vec<usize>, Vec<[f64; 2]>) -> P,
) -> Result<CollectedPaths<'g, P>, PathDomainFailure> {
    let mut budget = Budget {
        remaining: limits.max_work,
        deadline: limits.deadline,
        until_clock_check: 0,
    };
    budget.spend(
        graph
            .vertices
            .len()
            .checked_add(graph.arcs.len())
            .ok_or(PathDomainFailure::WorkLimit)?,
    )?;
    graph
        .validate()
        .map_err(|_| PathDomainFailure::InvalidGraph)?;
    budget.spend(0)?;
    let mut result = CollectedPaths {
        graph,
        paths: vec![],
        work_used: 0,
        stored_arcs: 0,
    };
    let mut initial = [0.; 2];
    for (r, value) in initial.iter_mut().enumerate().take(graph.resources.len()) {
        let bounds = graph.vertices[graph.source].bounds[r];
        *value = graph.initial[r].unwrap_or(bounds.lower);
        if *value < bounds.lower || *value > bounds.upper {
            budget.spend(0)?;
            result.work_used = limits.max_work - budget.remaining;
            return Ok(result);
        }
    }
    let n = graph.vertices.len();
    // Retain physical arc positions and their original order, including
    // parallel arcs. A DFS frame must scan only its outgoing arcs, rather
    // than the entire graph at every vertex of every partial path.
    budget.spend(
        n.checked_add(graph.arcs.len())
            .ok_or(PathDomainFailure::WorkLimit)?,
    )?;
    let mut outgoing = Vec::new();
    reserve(&mut outgoing, n)?;
    outgoing.resize_with(n, Vec::new);
    for (position, arc) in graph.arcs.iter().enumerate() {
        outgoing[arc.tail]
            .try_reserve(1)
            .map_err(|_| PathDomainFailure::Allocation)?;
        outgoing[arc.tail].push(position);
    }
    budget.spend(0)?;
    let mut visited = vec![];
    reserve(&mut visited, n)?;
    visited.resize(n, false);
    let mut stack = vec![];
    reserve(&mut stack, n)?;
    let mut prefix = vec![];
    reserve(&mut prefix, n)?;
    visited[graph.source] = true;
    stack.push(Frame {
        vertex: graph.source,
        next_arc: 0,
        resources: initial,
    });
    while !stack.is_empty() {
        budget.spend(1)?;
        let frame = stack.last_mut().unwrap();
        if frame.next_arc == outgoing[frame.vertex].len() {
            let frame = stack.pop().unwrap();
            visited[frame.vertex] = false;
            if !stack.is_empty() {
                prefix.pop();
            }
            continue;
        }
        let physical = outgoing[frame.vertex][frame.next_arc];
        frame.next_arc += 1;
        let arc = &graph.arcs[physical];
        if visited[arc.head] {
            continue;
        }
        let Some(resources) = extend(graph, arc, frame.resources)? else {
            continue;
        };
        if arc.head == graph.sink {
            if result.paths.len() == limits.max_paths {
                return Err(PathDomainFailure::PathLimit);
            }
            let length = prefix.len() + 1;
            let stored = result
                .stored_arcs
                .checked_add(length)
                .ok_or(PathDomainFailure::StoredArcLimit)?;
            if stored > limits.max_stored_arcs {
                return Err(PathDomainFailure::StoredArcLimit);
            }
            budget.spend(
                length
                    .checked_add(stack.len() + 1)
                    .ok_or(PathDomainFailure::WorkLimit)?,
            )?;
            let mut arcs = vec![];
            reserve(&mut arcs, length)?;
            arcs.extend_from_slice(&prefix);
            arcs.push(physical);
            let mut profile = vec![];
            reserve(&mut profile, stack.len() + 1)?;
            profile.extend(stack.iter().map(|frame| frame.resources));
            profile.push(resources);
            if result.paths.len() == result.paths.capacity() {
                let additional =
                    (limits.max_paths - result.paths.len()).min(result.paths.len().max(16));
                reserve(&mut result.paths, additional)?;
            }
            result.paths.push(make_path(arcs, profile));
            result.stored_arcs = stored;
        } else {
            visited[arc.head] = true;
            prefix.push(physical);
            stack.push(Frame {
                vertex: arc.head,
                next_arc: 0,
                resources,
            });
        }
    }
    budget.spend(0)?;
    result.work_used = limits.max_work - budget.remaining;
    Ok(result)
}
impl<'g> CompleteCanonicalPaths<'g> {
    /// Failure discards every candidate; only an exhausted search completes.
    /// Work counts validation size, scans and payload copy. Validation and
    /// allocation are cooperative checkpoints, not hard realtime.
    pub fn collect(graph: &'g Graph, limits: PathDomainLimits) -> Result<Self, PathDomainFailure> {
        let data = collect_paths(
            graph,
            limits,
            |g, arc, q| Ok(extend_resources(g, arc, q, true)),
            |arcs, resources| CanonicalPath { arcs, resources },
        )?;
        Ok(Self { data })
    }
    pub fn graph(&self) -> &'g Graph {
        self.data.graph
    }
    pub fn paths(&self) -> &[CanonicalPath] {
        &self.data.paths
    }
    pub fn work_used(&self) -> usize {
        self.data.work_used
    }
    pub fn stored_arcs(&self) -> usize {
        self.data.stored_arcs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ResourceKind;
    fn limits() -> PathDomainLimits {
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        }
    }
    fn graph() -> Graph {
        let mut graph = Graph::new(4, 1).unwrap();
        for v in 0..4 {
            graph.set_bounds(v, 0, 0., 5.).unwrap();
        }
        graph.set_bounds(1, 0, 2., 5.).unwrap();
        graph.set_bounds(2, 0, 0., 3.).unwrap();
        for (u, v, c) in [
            (0, 1, 0.),
            (0, 1, 0.),
            (1, 2, 2.),
            (2, 1, 0.),
            (1, 3, 1.),
            (2, 3, 1.),
            (0, 2, 1.),
        ] {
            let id = graph.add_arc(u, v, 1.).unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = c;
        }
        graph
    }
    #[test]
    fn parallel_arcs_cycles_waiting_and_equality_have_explicit_canonical_paths() {
        let graph = graph();
        let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
        let paths = domain
            .paths()
            .iter()
            .map(|p| p.arc_positions().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(paths, [vec![0, 4], vec![1, 4], vec![6, 3, 4], vec![6, 5]]);
        assert_eq!(
            domain.paths()[0].resource_profile(),
            &[[0., 0.], [2., 0.], [3., 0.]]
        );
        let mut equality = graph.clone();
        equality.resources[0] = ResourceKind::Equality;
        let domain = CompleteCanonicalPaths::collect(&equality, limits()).unwrap();
        assert_eq!(domain.paths().len(), 1);
        assert_eq!(domain.paths()[0].arc_positions(), &[6, 5]);
    }
    #[test]
    fn limits_reject_partial_results_and_accept_exact_boundaries() {
        let graph = graph();
        let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
        let exact = PathDomainLimits {
            max_paths: domain.paths().len(),
            max_stored_arcs: domain.stored_arcs(),
            max_work: domain.work_used(),
            deadline: None,
        };
        assert!(CompleteCanonicalPaths::collect(&graph, exact).is_ok());
        for (limit, expected) in [
            (
                PathDomainLimits {
                    max_paths: exact.max_paths - 1,
                    ..exact
                },
                PathDomainFailure::PathLimit,
            ),
            (
                PathDomainLimits {
                    max_stored_arcs: exact.max_stored_arcs - 1,
                    ..exact
                },
                PathDomainFailure::StoredArcLimit,
            ),
            (
                PathDomainLimits {
                    max_work: exact.max_work - 1,
                    ..exact
                },
                PathDomainFailure::WorkLimit,
            ),
            (
                PathDomainLimits {
                    deadline: Some(Instant::now()),
                    ..exact
                },
                PathDomainFailure::TimeLimit,
            ),
        ] {
            assert_eq!(
                CompleteCanonicalPaths::collect(&graph, limit).unwrap_err(),
                expected
            );
        }
    }
    #[test]
    fn empty_domain_and_impossible_initial_state_are_complete() {
        let graph = Graph::new(2, 0).unwrap();
        let limit = PathDomainLimits {
            max_paths: 0,
            max_stored_arcs: 0,
            ..limits()
        };
        assert!(CompleteCanonicalPaths::collect(&graph, limit)
            .unwrap()
            .paths()
            .is_empty());
        let mut graph = super::tests::graph();
        graph.initial[0] = Some(10.);
        assert!(CompleteCanonicalPaths::collect(&graph, limits())
            .unwrap()
            .paths()
            .is_empty());
    }
    #[test]
    fn direct_paths_and_cost_overflow_are_not_silently_omitted() {
        let mut graph = Graph::new(2, 0).unwrap();
        graph.add_arc(0, 1, 1.).unwrap();
        let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
        assert_eq!(domain.paths()[0].arc_positions(), &[0]);
        let mut graph = Graph::new(3, 0).unwrap();
        graph.add_arc(0, 1, f64::MAX).unwrap();
        graph.add_arc(1, 2, f64::MAX).unwrap();
        let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
        assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
    }
    #[test]
    fn malformed_graphs_are_not_completed_domains() {
        let mut graph = graph();
        graph.arcs[0].head = 99;
        assert_eq!(
            CompleteCanonicalPaths::collect(&graph, limits()).unwrap_err(),
            PathDomainFailure::InvalidGraph
        );
    }

    #[test]
    fn sparse_path_search_does_not_rescan_unrelated_arcs() {
        let mut graph = Graph::new(80, 0).unwrap();
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 79, 1.).unwrap();
        for tail in 2..79 {
            for head in 2..79 {
                if tail != head {
                    graph.add_arc(tail, head, 1.).unwrap();
                }
            }
        }
        // Validation and adjacency construction are linear in graph size.
        // The reachable DFS has only two arcs, despite the dense unreachable
        // component. A full-graph scan at each frame exceeds this budget.
        let limits = PathDomainLimits {
            max_work: 2 * (graph.vertices.len() + graph.arcs.len()) + 16,
            deadline: None,
            ..limits()
        };
        let domain = CompleteCanonicalPaths::collect(&graph, limits).unwrap();
        assert_eq!(domain.paths().len(), 1);
        assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
    }

    fn chain(consumption: &[f64], upper: f64) -> Graph {
        let mut g = Graph::new(consumption.len() + 1, 1).unwrap();
        g.initial[0] = Some(0.);
        for vertex in 0..g.vertices.len() {
            g.set_bounds(vertex, 0, 0., upper).unwrap();
        }
        for (i, &value) in consumption.iter().enumerate() {
            let id = g.add_arc(i, i + 1, 1.).unwrap();
            g.arc_mut(id).unwrap().consumption[0] = value;
        }
        g
    }
    #[test]
    fn directed_relaxation_retains_an_exact_real_feasible_rounding_boundary() {
        // The exact sum is 1 + 3 ulps. Two nearest/even additions of 1.5 ulps
        // round upward, producing 1 + 4 ulps and a spurious window rejection.
        let delta = 3. * 2f64.powi(-53);
        let upper = f64::from_bits(1f64.to_bits() + 3);
        let g = chain(&[1., delta, delta], upper);
        assert!(CompleteCanonicalPaths::collect(&g, limits())
            .unwrap()
            .paths()
            .is_empty());
        let domain = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
        assert_eq!(domain.paths().len(), 1);
        assert_eq!(domain.paths()[0].arc_positions(), &[0, 1, 2]);
        assert!(domain.paths()[0].resource_lower_bounds()[3][0] <= upper);
    }
    #[test]
    fn relaxation_may_retain_infeasible_paths_but_rejects_proven_violations() {
        let g = chain(&[0.7, 0.3], f64::from_bits(1f64.to_bits() - 1));
        assert_eq!(
            CompleteResourceRelaxation::collect(&g, limits())
                .unwrap()
                .paths()
                .len(),
            1
        );
        let g = chain(&[0.7, 0.4], 1.);
        assert!(CompleteResourceRelaxation::collect(&g, limits())
            .unwrap()
            .paths()
            .is_empty());
        let g = graph();
        let a = CompleteCanonicalPaths::collect(&g, limits()).unwrap();
        let b = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
        assert_eq!(
            a.paths()
                .iter()
                .map(|p| p.arc_positions())
                .collect::<Vec<_>>(),
            b.paths()
                .iter()
                .map(|p| p.arc_positions())
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn relaxed_collection_has_atomic_limits_and_explicit_unknown_arithmetic() {
        let g = chain(&[1., 0.5], 2.);
        let domain = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
        let exact = PathDomainLimits {
            max_paths: 1,
            max_stored_arcs: 2,
            max_work: domain.work_used(),
            deadline: None,
        };
        assert!(CompleteResourceRelaxation::collect(&g, exact).is_ok());
        for (l, want) in [
            (
                PathDomainLimits {
                    max_paths: 0,
                    ..exact
                },
                PathDomainFailure::PathLimit,
            ),
            (
                PathDomainLimits {
                    max_stored_arcs: 1,
                    ..exact
                },
                PathDomainFailure::StoredArcLimit,
            ),
            (
                PathDomainLimits {
                    max_work: exact.max_work - 1,
                    ..exact
                },
                PathDomainFailure::WorkLimit,
            ),
            (
                PathDomainLimits {
                    deadline: Some(Instant::now()),
                    ..exact
                },
                PathDomainFailure::TimeLimit,
            ),
        ] {
            assert_eq!(
                CompleteResourceRelaxation::collect(&g, l).unwrap_err(),
                want
            );
        }
        let g = chain(&[f64::MAX, f64::MAX], f64::MAX);
        assert_eq!(
            CompleteResourceRelaxation::collect(&g, limits()).unwrap_err(),
            PathDomainFailure::Arithmetic
        );
        let mut g = chain(&[1., 1.], 2.);
        g.resources[0] = ResourceKind::Equality;
        assert_eq!(
            CompleteResourceRelaxation::collect(&g, limits()).unwrap_err(),
            PathDomainFailure::UnsupportedResources
        );
    }
}
