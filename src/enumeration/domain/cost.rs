//! Directed additive cost pruning for physical-simple resource relaxations.
use super::*;
use crate::core::rounding::add_down;

/// All exact-resource-feasible physical-simple paths whose additive arc cost is
/// at most `ceiling` are included. Arc costs are caller-supplied lower bounds;
/// extra paths can remain because resource and cost arithmetic round downward.
/// This is cutoff coverage, never a complete original-domain certificate.
#[derive(Debug)]
pub struct CostBoundedResourceRelaxation<
    'g,
    const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS },
> {
    data: CollectedPaths<'g, RelaxedResourcePath<D>, D>,
    ceiling: f64,
    arc_cost_lower_bounds: &'g [f64],
}
impl<'g, const D: usize> CostBoundedResourceRelaxation<'g, D> {
    /// `arc_cost_lower_bounds` uses physical arc positions, including parallel
    /// arcs. The caller must prove these bounds against its intended objective
    /// or reduced costs; this collector establishes only their additive scope.
    pub fn collect(
        graph: &'g Graph<D>,
        arc_cost_lower_bounds: &'g [f64],
        ceiling: f64,
        limits: PathDomainLimits,
    ) -> Result<Self, PathDomainFailure> {
        if graph
            .resources
            .iter()
            .any(|r| *r != crate::graph::ResourceKind::Disposable)
        {
            return Err(PathDomainFailure::UnsupportedResources);
        }
        let data = collect_paths_with_clock_and_cost(
            graph,
            limits,
            extend_resource_lower_bounds,
            |arcs, lower_resources| RelaxedResourcePath {
                arcs,
                lower_resources,
            },
            Instant::now,
            Some((arc_cost_lower_bounds, ceiling)),
        )?;
        Ok(Self {
            data,
            ceiling,
            arc_cost_lower_bounds,
        })
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
    /// The immutable physical-arc weights to which coverage applies.
    pub fn arc_cost_lower_bounds(&self) -> &[f64] {
        self.arc_cost_lower_bounds
    }
    pub fn coverage_ceiling(&self) -> f64 {
        self.ceiling
    }
}

pub(super) struct CostPruning<'a> {
    costs: &'a [f64],
    suffix: Vec<Option<f64>>,
    ceiling: f64,
    sink: usize,
}
impl<'a> CostPruning<'a> {
    pub(super) fn new<const D: usize, C: FnMut() -> Instant>(
        graph: &Graph<D>,
        costs: &'a [f64],
        ceiling: f64,
        budget: &mut Budget<C>,
    ) -> Result<Self, PathDomainFailure> {
        if costs.len() != graph.arcs.len()
            || !ceiling.is_finite()
            || costs.iter().any(|c| !c.is_finite())
        {
            return Err(PathDomainFailure::Arithmetic);
        }
        let n = graph.vertices.len();
        budget.spend(n)?;
        let mut suffix = vec![];
        reserve(&mut suffix, n)?;
        suffix.resize(n, None);
        suffix[graph.sink] = Some(0.);
        // Synchronous at-most-k walk relaxation. Every simple suffix has at
        // most n-1 arcs. Negative cycles only loosen this finite lower bound;
        // no nonnegative-cost or acyclic-graph assumption is needed.
        for _ in 1..n {
            budget.spend(n)?;
            let mut next = vec![];
            reserve(&mut next, n)?;
            next.extend_from_slice(&suffix);
            for (arc, &cost) in graph.arcs.iter().zip(costs) {
                budget.spend(1)?;
                if let Some(tail) = suffix[arc.head] {
                    let value = add_down(cost, tail).ok_or(PathDomainFailure::Arithmetic)?;
                    if next[arc.tail].map_or(true, |old| value < old) {
                        next[arc.tail] = Some(value);
                    }
                }
            }
            suffix = next;
        }
        Ok(Self {
            costs,
            suffix,
            ceiling,
            sink: graph.sink,
        })
    }
    pub(super) fn permits<C: FnMut() -> Instant>(
        &self,
        prefix: &[usize],
        physical: usize,
        head: usize,
        budget: &mut Budget<C>,
    ) -> Result<bool, PathDomainFailure> {
        budget.spend(
            prefix
                .len()
                .checked_add(1)
                .ok_or(PathDomainFailure::WorkLimit)?,
        )?;
        let mut lower = 0.;
        for &arc in prefix.iter().chain(std::iter::once(&physical)) {
            lower = add_down(lower, self.costs[arc]).ok_or(PathDomainFailure::Arithmetic)?;
        }
        if head != self.sink {
            let Some(suffix) = self.suffix[head] else {
                return Ok(false);
            };
            lower = add_down(lower, suffix).ok_or(PathDomainFailure::Arithmetic)?;
        }
        Ok(lower <= self.ceiling)
    }
}
