//! Complete physical-simple paths or explicitly bounded walks under scalar resources.
//! No reduced-cost cutoff, bucket fixing, NG, cut or branch restriction is used.
use crate::graph::Graph;
use crate::labeling::extend_resources;
use std::time::{Duration, Instant};

mod cost;
pub use cost::CostBoundedResourceRelaxation;

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
pub struct CanonicalPath<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    arcs: Vec<usize>,
    resources: Vec<crate::graph::ResourceValues<D>>,
}
impl<const D: usize> CanonicalPath<D> {
    /// Positions in the borrowed original graph, retaining parallel arc identity.
    pub fn arc_positions(&self) -> &[usize] {
        &self.arcs
    }
    /// Source state and state after each arc; unused resource slots are zero.
    pub fn resource_profile(&self) -> &[crate::graph::ResourceValues<D>] {
        &self.resources
    }
}

/// A physical path with downward resource bounds, rather than a feasible
/// floating-point schedule. It must not be submitted as an integer witness.
#[derive(Debug)]
pub struct RelaxedResourcePath<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    arcs: Vec<usize>,
    lower_resources: Vec<crate::graph::ResourceValues<D>>,
}
impl<const D: usize> RelaxedResourcePath<D> {
    pub fn arc_positions(&self) -> &[usize] {
        &self.arcs
    }
    /// Source value and a lower bound after each arc on every exact-real
    /// feasible schedule along this path; unused slots are zero.
    pub fn resource_lower_bounds(&self) -> &[crate::graph::ResourceValues<D>] {
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
pub struct CompleteResourceRelaxation<'g, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    data: CollectedPaths<'g, RelaxedResourcePath<D>, D>,
}
impl<'g, const D: usize> CompleteResourceRelaxation<'g, D> {
    /// The same path/work/storage/deadline controls as canonical collection.
    /// Unsupported kinds or arithmetic failure return no partial universe.
    pub fn collect(
        graph: &'g Graph<D>,
        limits: PathDomainLimits,
    ) -> Result<Self, PathDomainFailure> {
        if limits.deadline.is_some_and(|end| Instant::now() >= end) {
            return Err(PathDomainFailure::TimeLimit);
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
            false,
            extend_resource_lower_bounds,
            |arcs, lower_resources| RelaxedResourcePath {
                arcs,
                lower_resources,
            },
        )?;
        Ok(Self { data })
    }
    pub fn graph(&self) -> &'g Graph<D> {
        self.data.graph
    }
    pub fn paths(&self) -> &[RelaxedResourcePath<D>] {
        &self.data.paths
    }
    pub fn work_used(&self) -> usize {
        self.data.work_used
    }
    pub fn stored_arcs(&self) -> usize {
        self.data.stored_arcs
    }
}
fn extend_resource_lower_bounds<const D: usize>(
    g: &Graph<D>,
    arc: &crate::graph::Arc<D>,
    mut q: crate::graph::ResourceValues<D>,
) -> Result<Option<crate::graph::ResourceValues<D>>, PathDomainFailure> {
    for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
        let mut bounds = g.vertices[arc.head].bounds[r];
        if let Some(window) = arc.bounds[r] {
            bounds.lower = bounds.lower.max(window.lower);
            bounds.upper = bounds.upper.min(window.upper);
        }
        *value = crate::core::rounding::add_down(*value, arc.consumption[r])
            .ok_or(PathDomainFailure::Arithmetic)?
            .max(bounds.lower);
        if *value > bounds.upper {
            return Ok(None);
        }
    }
    Ok(Some(q))
}

/// Every route-domain and elementarity-valid source/sink path accepted by
/// canonical forward resource extension, including a direct source/sink arc.
/// One earliest profile is kept.
/// This is not a proof that all feasible resource schedules or original routing
/// columns are represented. Such equivalence must be established by the caller.
/// The graph borrow prevents stale membership after graph mutation.
///
/// ```compile_fail
/// use baldes::{Graph, enumeration::{CompleteCanonicalPaths, PathDomainLimits}};
/// let mut graph = Graph::<2>::new(2, 0).unwrap();
/// let domain = CompleteCanonicalPaths::collect(&graph, PathDomainLimits::default()).unwrap();
/// graph.arcs.clear();
/// assert!(domain.paths().is_empty());
/// ```
#[derive(Debug)]
pub struct CompleteCanonicalPaths<'g, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    data: CollectedPaths<'g, CanonicalPath<D>, D>,
}
#[derive(Debug)]
struct CollectedPaths<'g, P, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    graph: &'g Graph<D>,
    paths: Vec<P>,
    work_used: usize,
    stored_arcs: usize,
}
struct Budget<C> {
    remaining: usize,
    deadline: Option<Instant>,
    until_clock_check: usize,
    clock: C,
}
impl<C: FnMut() -> Instant> Budget<C> {
    fn spend(&mut self, work: usize) -> Result<(), PathDomainFailure> {
        // Collection is cooperatively timed. Poll once per small batch of
        // work, and always at explicit validation/completion checkpoints.
        if work == 0 || work >= self.until_clock_check {
            if self.deadline.is_some_and(|end| (self.clock)() >= end) {
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
struct Frame<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    vertex: usize,
    next_arc: usize,
    resources: crate::graph::ResourceValues<D>,
    elementarity_len: usize,
}
fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), PathDomainFailure> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| PathDomainFailure::Allocation)
}
fn collect_paths<'g, P, const D: usize>(
    graph: &'g Graph<D>,
    limits: PathDomainLimits,
    respect_elementarity: bool,
    extend: impl FnMut(
        &Graph<D>,
        &crate::graph::Arc<D>,
        crate::graph::ResourceValues<D>,
    ) -> Result<Option<crate::graph::ResourceValues<D>>, PathDomainFailure>,
    make_path: impl FnMut(Vec<usize>, Vec<crate::graph::ResourceValues<D>>) -> P,
) -> Result<CollectedPaths<'g, P, D>, PathDomainFailure> {
    // These collectors certify a scalar relaxation with additive costs. Until
    // coupled interval costs are carried by that relaxation, report unsupported.
    if graph.has_coupled_resources() {
        return Err(PathDomainFailure::UnsupportedResources);
    }
    collect_paths_with_clock(
        graph,
        limits,
        respect_elementarity,
        extend,
        make_path,
        Instant::now,
    )
}

fn collect_paths_with_clock<'g, P, const D: usize>(
    graph: &'g Graph<D>,
    limits: PathDomainLimits,
    respect_elementarity: bool,
    extend: impl FnMut(
        &Graph<D>,
        &crate::graph::Arc<D>,
        crate::graph::ResourceValues<D>,
    ) -> Result<Option<crate::graph::ResourceValues<D>>, PathDomainFailure>,
    make_path: impl FnMut(Vec<usize>, Vec<crate::graph::ResourceValues<D>>) -> P,
    clock: impl FnMut() -> Instant,
) -> Result<CollectedPaths<'g, P, D>, PathDomainFailure> {
    collect_paths_with_clock_and_cost(
        graph,
        limits,
        respect_elementarity,
        extend,
        make_path,
        clock,
        None,
    )
}

fn collect_paths_with_clock_and_cost<'g, P, const D: usize>(
    graph: &'g Graph<D>,
    limits: PathDomainLimits,
    respect_elementarity: bool,
    mut extend: impl FnMut(
        &Graph<D>,
        &crate::graph::Arc<D>,
        crate::graph::ResourceValues<D>,
    ) -> Result<Option<crate::graph::ResourceValues<D>>, PathDomainFailure>,
    mut make_path: impl FnMut(Vec<usize>, Vec<crate::graph::ResourceValues<D>>) -> P,
    clock: impl FnMut() -> Instant,
    cost_bounds: Option<(&[f64], f64)>,
) -> Result<CollectedPaths<'g, P, D>, PathDomainFailure> {
    if graph.has_coupled_resources() {
        return Err(PathDomainFailure::UnsupportedResources);
    }
    let mut budget = Budget {
        remaining: limits.max_work,
        deadline: limits.deadline,
        until_clock_check: 0,
        clock,
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
    let mut initial = crate::graph::ResourceValues::filled(graph.resources.len(), 0.);
    for (r, value) in initial.iter_mut().enumerate().take(graph.resources.len()) {
        let bounds = graph.vertices[graph.source].bounds[r];
        let Some(root) = graph.root_resource(r, true) else {
            result.work_used = limits.max_work - budget.remaining;
            return Ok(result);
        };
        *value = root;
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
    let cost_pruning = cost_bounds
        .map(|(costs, ceiling)| cost::CostPruning::new(graph, costs, ceiling, &mut budget))
        .transpose()?;
    let mut visited = vec![];
    reserve(&mut visited, n)?;
    visited.resize(n, false);
    let enforce_groups = respect_elementarity && graph.has_explicit_elementarity_groups();
    let mut used_elements = vec![];
    let mut element_path = vec![];
    if enforce_groups {
        budget.spend(graph.element_count)?;
        reserve(&mut used_elements, graph.element_count)?;
        used_elements.resize(graph.element_count, false);
        reserve(&mut element_path, graph.element_count)?;
    }
    let source_elements = if enforce_groups {
        graph.vertices[graph.source].elements.as_slice()
    } else {
        &[]
    };
    for &element in source_elements {
        if element >= used_elements.len() || used_elements[element] {
            return Err(PathDomainFailure::InvalidGraph);
        }
        used_elements[element] = true;
        element_path.push(element);
    }
    let mut stack = vec![];
    reserve(&mut stack, n)?;
    let mut prefix = vec![];
    reserve(&mut prefix, n)?;
    visited[graph.source] = true;
    stack.push(Frame {
        vertex: graph.source,
        next_arc: 0,
        resources: initial,
        elementarity_len: element_path.len(),
    });
    while !stack.is_empty() {
        budget.spend(1)?;
        let frame = stack.last_mut().unwrap();
        if frame.next_arc == outgoing[frame.vertex].len()
            || prefix.len() >= graph.original_max_arcs()
        {
            let frame = stack.pop().unwrap();
            while element_path.len() > frame.elementarity_len {
                used_elements[element_path.pop().unwrap()] = false;
            }
            visited[frame.vertex] = false;
            if !stack.is_empty() {
                prefix.pop();
            }
            continue;
        }
        let physical = outgoing[frame.vertex][frame.next_arc];
        frame.next_arc += 1;
        let arc = &graph.arcs[physical];
        if arc.head == graph.source || (graph.max_route_arcs().is_none() && visited[arc.head]) {
            continue;
        }
        if let Some(pruning) = &cost_pruning {
            if !pruning.permits(&prefix, physical, arc.head, &mut budget)? {
                continue;
            }
        }
        let Some(resources) = extend(graph, arc, frame.resources)? else {
            continue;
        };
        let elementarity_len = element_path.len();
        if enforce_groups {
            budget.spend(
                arc.elements
                    .len()
                    .checked_add(graph.vertices[arc.head].elements.len())
                    .ok_or(PathDomainFailure::WorkLimit)?,
            )?;
            let mut conflict = false;
            for &element in arc
                .elements
                .iter()
                .chain(&graph.vertices[arc.head].elements)
            {
                if element >= used_elements.len() {
                    conflict = true;
                    break;
                }
                if element_path[elementarity_len..].contains(&element) {
                    continue;
                }
                if used_elements[element] {
                    conflict = true;
                    break;
                }
                used_elements[element] = true;
                element_path.push(element);
            }
            if conflict {
                while element_path.len() > elementarity_len {
                    used_elements[element_path.pop().unwrap()] = false;
                }
                continue;
            }
        }
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
            while element_path.len() > elementarity_len {
                used_elements[element_path.pop().unwrap()] = false;
            }
        } else {
            visited[arc.head] = true;
            if prefix.len() == prefix.capacity() {
                reserve(&mut prefix, 1)?;
            }
            if stack.len() == stack.capacity() {
                reserve(&mut stack, 1)?;
            }
            prefix.push(physical);
            stack.push(Frame {
                vertex: arc.head,
                next_arc: 0,
                resources,
                elementarity_len,
            });
        }
    }
    budget.spend(0)?;
    result.work_used = limits.max_work - budget.remaining;
    Ok(result)
}
impl<'g, const D: usize> CompleteCanonicalPaths<'g, D> {
    /// Failure discards every candidate; only an exhausted search completes.
    /// Work counts validation size, scans and payload copy. Validation and
    /// allocation are cooperative checkpoints, not hard realtime.
    pub fn collect(
        graph: &'g Graph<D>,
        limits: PathDomainLimits,
    ) -> Result<Self, PathDomainFailure> {
        let data = collect_paths(
            graph,
            limits,
            true,
            |g, arc, q| Ok(extend_resources(g, arc, &q, true)),
            |arcs, resources| CanonicalPath { arcs, resources },
        )?;
        Ok(Self { data })
    }
    pub fn graph(&self) -> &'g Graph<D> {
        self.data.graph
    }
    pub fn paths(&self) -> &[CanonicalPath<D>] {
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
#[path = "../../../tests/unit/enumeration/domain/tests.rs"]
mod tests;
