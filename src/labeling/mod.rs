//! Forward no-cut labeling with immutable predecessor identities.
mod arena;
pub mod engine;
pub mod pool;
mod resources;
pub mod storage;
use crate::graph::ng::{NgMask, PreparedNg};
use crate::graph::{Arc, Graph, ResourceKind};
use crate::{incomplete, invalid, Result};
pub use resources::ScalarResourceKernel;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const EPSILON: f64 = 1e-6;
pub const NO_ROUTE_COST: f64 = 1e12;
/// Equality-resource dominance uses rounded interval endpoints. Subtracting
/// the states first (`abs(a-b)`) changes decisions at representable boundaries.
#[inline(always)]
pub(crate) fn equality_resource_dominates(a: f64, b: f64) -> bool {
    b - EPSILON <= a && a <= b + EPSILON
}
#[derive(Clone, Debug)]
pub struct Controls {
    pub max_labels: usize,
    pub max_columns: usize,
    pub max_route_arcs: Option<usize>,
    pub time_limit: Duration,
    /// Output filtering only; never prunes a prefix with negative future arcs.
    pub negative_only: bool,
    pub zero_threshold: f64,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            // No label limit: R202 with rank-one cuts exceeds a
            // million labels in one pricing call.
            max_labels: usize::MAX,
            max_columns: 150,
            max_route_arcs: None,
            time_limit: Duration::from_secs(1000),
            negative_only: false,
            zero_threshold: 0.0,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub nodes: Vec<i32>,
    pub arcs: Vec<i32>,
    pub cost: f64,
    pub original_cost: f64,
    pub resources: Vec<f64>,
    /// Resource states at the source and after each arc, including waiting
    /// introduced by bucket jumps. Distinct schedules can share an arc path.
    pub resource_profile: Vec<Vec<f64>>,
}
impl Route {
    pub(crate) fn identity(&self) -> String {
        use std::fmt::Write;
        let capacity =
            self.resource_profile
                .iter()
                .fold(self.arcs.len().saturating_mul(12), |size, state| {
                    size.saturating_add(1)
                        .saturating_add(state.len().saturating_mul(17))
                });
        let mut identity = String::with_capacity(capacity);
        for (i, arc) in self.arcs.iter().enumerate() {
            if i != 0 {
                identity.push(',');
            }
            write!(&mut identity, "{arc}").unwrap();
        }
        for state in &self.resource_profile {
            identity.push('|');
            for &value in state {
                // Native vector ordering treats signed zeros as equal.
                let bits = if value == 0. { 0 } else { value.to_bits() };
                write!(&mut identity, "{bits:016x},").unwrap();
            }
        }
        identity
    }
}
#[cfg(test)]
#[path = "../../tests/unit/labeling/identity_tests.rs"]
mod identity_tests;
#[derive(Clone, Debug, Default)]
pub struct Statistics {
    pub generated: usize,
    pub expanded: usize,
    pub dominated: usize,
    pub dominance_checks: usize,
    pub other_dominance_checks: usize,
    pub other_bucket_searches: usize,
    pub concatenation_tests: usize,
    pub later_label_checks: usize,
    pub other_dominated: usize,
    pub capped: usize,
    pub forward_labels: usize,
    pub backward_labels: usize,
    pub elapsed: f64,
    pub forward_time: f64,
    pub backward_time: f64,
    pub concatenation_time: f64,
}
impl Statistics {
    /// Adds another sweep's counters; the times are left as they are.
    pub fn absorb(&mut self, other: &Statistics) {
        let Statistics {
            generated,
            expanded,
            dominated,
            dominance_checks,
            other_dominance_checks,
            other_bucket_searches,
            concatenation_tests,
            later_label_checks,
            other_dominated,
            capped,
            forward_labels,
            backward_labels,
            elapsed: _,
            forward_time: _,
            backward_time: _,
            concatenation_time: _,
        } = other;
        self.generated += generated;
        self.expanded += expanded;
        self.dominated += dominated;
        self.dominance_checks += dominance_checks;
        self.other_dominance_checks += other_dominance_checks;
        self.other_bucket_searches += other_bucket_searches;
        self.concatenation_tests += concatenation_tests;
        self.later_label_checks += later_label_checks;
        self.other_dominated += other_dominated;
        self.capped += capped;
        self.forward_labels += forward_labels;
        self.backward_labels += backward_labels;
    }
}
#[derive(Debug)]
pub struct PricingResult {
    pub columns: Vec<Route>,
    pub minimum_reduced_cost: f64,
    pub exact: bool,
    pub statistics: Statistics,
}
#[derive(Clone, Debug)]
struct Label<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    coupled_state: Option<Box<crate::graph::coupled::CoupledLabelState>>,
    vertex: usize,
    cost: f64,
    original_cost: f64,
    resources: crate::graph::ResourceValues<D>,
    ng: NgMask,
    parent: Option<usize>,
    arc: Option<usize>,
    hops: usize,
    active: bool,
}

/// StdResources transitions: disposable resources may wait at a lower bound;
/// equality resources cannot. Backward propagation clips before subtraction.
/// Each resource goes through a scalar step so `q` stays in registers; a
/// loop writing `q[r]` kept it in memory, and the next 16-byte read of the
/// resources stalled on store forwarding.
#[inline(always)]
pub fn extend_resources<const D: usize>(
    g: &Graph<D>,
    a: &Arc<D>,
    q: impl AsRef<[f64]>,
    forward: bool,
) -> Option<crate::graph::ResourceValues<D>> {
    let q = q.as_ref();
    if g.resources.len() > D || q.len() < g.resources.len() {
        return None;
    }
    let values = std::array::from_fn(|r| q.get(r).copied().unwrap_or(0.));
    ScalarResourceKernel::<D>::extend_active(g, a, values, forward).map(Into::into)
}
#[inline(always)]
pub(crate) fn extend_resource<const D: usize>(
    g: &Graph<D>,
    a: &Arc<D>,
    r: usize,
    q: f64,
    forward: bool,
) -> Option<f64> {
    let mut head = g.vertices[a.head].bounds[r];
    if let Some(window) = a.bounds[r] {
        head.lower = head.lower.max(window.lower);
        head.upper = head.upper.min(window.upper);
    }
    if forward {
        g.project_resource(a.head, r, head, q + a.consumption[r], true)
    } else {
        let mut tail = g.vertices[a.tail].bounds[r];
        if a.tail == g.source {
            if let Some(initial) = g.initial[r] {
                tail.lower = tail.lower.max(initial);
            }
        }
        let head_value = g.project_resource(a.head, r, head, q, false)?;
        g.project_resource(a.tail, r, tail, head_value - a.consumption[r], false)
    }
}

fn dominates<const D: usize>(g: &Graph<D>, a: &Label<D>, b: &Label<D>, hop_limit: bool) -> bool {
    // This reference label type has no cut/additional-cost component. The
    // coupled gate below checks independently tracked scalar/resource order;
    // the cut-bearing engine additionally requires identical cut words.
    let Some(penalty) = g.coupled_dominance_penalty(
        a.coupled_state.as_deref(),
        b.coupled_state.as_deref(),
        true,
        a.vertex,
    ) else {
        return false;
    };
    let cost = if penalty == 0. {
        a.cost
    } else {
        crate::core::rounding::add_up(a.cost, penalty).unwrap_or(f64::INFINITY)
    };
    if a.vertex != b.vertex
        || cost - b.cost >= EPSILON
        || a.ng & !b.ng != 0
        || (hop_limit && a.hops > b.hops)
    {
        return false;
    }
    g.resources.iter().enumerate().all(|(r, k)| match k {
        ResourceKind::Disposable => a.resources[r] <= b.resources[r] + EPSILON,
        ResourceKind::Equality => equality_resource_dominates(a.resources[r], b.resources[r]),
    })
}
fn reconstruct<const D: usize>(g: &Graph<D>, labels: &[Label<D>], terminal: &Label<D>) -> Route {
    let mut arcs = vec![];
    let mut resource_profile = vec![];
    let mut next = Some(terminal);
    while let Some(l) = next {
        resource_profile.push(l.resources[..g.resources.len()].to_vec());
        if let Some(a) = l.arc {
            arcs.push(a);
        }
        next = l.parent.map(|id| &labels[id]);
    }
    arcs.reverse();
    resource_profile.reverse();
    let mut nodes = vec![g.source as i32];
    for &a in &arcs {
        nodes.push(g.arcs[a].head as i32);
    }
    Route {
        nodes,
        arcs: arcs.iter().map(|&a| g.arcs[a].id).collect(),
        cost: terminal.cost,
        original_cost: terminal.original_cost,
        resources: terminal.resources[..g.resources.len()].to_vec(),
        resource_profile,
    }
}

/// Complete a label-correcting search or return an error. Budget/time exhaustion
/// never returns a partial pool with an exactness certificate.
pub fn reference_price<const D: usize>(g: &Graph<D>, controls: &Controls) -> Result<PricingResult> {
    g.validate()?;
    if controls.max_labels == 0 || controls.max_columns == 0 || !controls.zero_threshold.is_finite()
    {
        return Err(invalid(
            "positive label/column limits and finite zero threshold required",
        ));
    }
    let ng = PreparedNg::new(g)?;
    let started = Instant::now();
    if controls.time_limit.is_zero() {
        return Err(incomplete("pricing hard time limit exceeded"));
    }
    let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
    for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
        *value = g
            .root_resource(r, true)
            .ok_or_else(|| invalid("initial resource outside source windows"))?;
        let w = g.vertices[g.source].bounds[r];
        if *value < w.lower || *value > w.upper {
            return Err(invalid("initial resource outside source window"));
        }
    }
    let root = Label {
        coupled_state: g.coupled_kernel().map(|k| k.root(true)),
        vertex: g.source,
        cost: 0.,
        original_cost: 0.,
        resources: q,
        ng: ng.vertex_bits[g.source],
        parent: None,
        arc: None,
        hops: 0,
        active: true,
    };
    let mut labels = vec![root];
    let mut by_vertex = vec![vec![]; g.vertices.len()];
    by_vertex[g.source].push(0);
    let mut queue = VecDeque::from([0usize]);
    let mut pool: Vec<Route> = vec![];
    let mut stats = Statistics {
        generated: 1,
        ..Statistics::default()
    };
    let mut outgoing = vec![vec![]; g.vertices.len()];
    for (i, a) in g.arcs.iter().enumerate() {
        outgoing[a.tail].push(i);
    }
    // Native graph order is head index after source/descending-customer/sink renumbering.
    let rank = |v: usize| {
        if v == g.source {
            0
        } else if v == g.sink {
            g.vertices.len() - 1
        } else {
            1 + (0..g.vertices.len())
                .filter(|&x| x != g.source && x != g.sink && x > v)
                .count()
        }
    };
    for arcs in &mut outgoing {
        arcs.sort_by_key(|&a| (rank(g.arcs[a].head), g.arcs[a].id));
    }
    let mut minimum = NO_ROUTE_COST;
    while let Some(id) = queue.pop_front() {
        if started.elapsed() >= controls.time_limit {
            return Err(incomplete("pricing hard time limit exceeded"));
        }
        if !labels[id].active {
            continue;
        }
        let from = labels[id].clone();
        stats.expanded += 1;
        if controls
            .max_route_arcs
            .is_some_and(|limit| from.hops >= limit)
        {
            continue;
        }
        for &index in &outgoing[from.vertex] {
            let a = &g.arcs[index];
            if a.head == g.source {
                continue;
            }
            let Some(resources) = extend_resources(g, a, &from.resources, true) else {
                continue;
            };
            let Some(memory) = ng.forward[index].extend(from.ng, ng.vertex_bits[a.head]) else {
                continue;
            };
            let (coupled_state, cost, original_cost) = if let Some(kernel) = g.coupled_kernel() {
                let state = from
                    .coupled_state
                    .as_deref()
                    .ok_or_else(|| invalid("missing coupled reference state"))?;
                let Some((mut next_state, _)) = kernel.extend(state, index, true)? else {
                    continue;
                };
                let (cost, original_cost) =
                    next_state.account_extension(state, a.reduced_cost, a.cost)?;
                (Some(next_state), cost, original_cost)
            } else {
                (
                    None,
                    from.cost + a.reduced_cost,
                    from.original_cost + a.cost,
                )
            };
            let next = Label {
                coupled_state,
                vertex: a.head,
                cost,
                original_cost,
                resources,
                ng: memory,
                parent: Some(id),
                arc: Some(index),
                hops: from.hops + 1,
                active: true,
            };
            if !next.cost.is_finite() || !next.original_cost.is_finite() {
                return Err(incomplete("route cost overflow"));
            }
            if stats.generated >= controls.max_labels {
                return Err(incomplete("pricing label limit exceeded"));
            }
            stats.generated += 1;
            if a.head == g.sink {
                minimum = minimum.min(next.cost);
                if !controls.negative_only || next.cost < controls.zero_threshold - EPSILON {
                    let route = reconstruct(g, &labels, &next);
                    // Retain a bounded pool, stable in native (cost, insertion) order.
                    let pos = pool.partition_point(|old| old.cost <= route.cost);
                    if pos < controls.max_columns {
                        pool.insert(pos, route);
                        if pool.len() > controls.max_columns {
                            pool.pop();
                        }
                    }
                }
                continue;
            }
            let previous = &mut by_vertex[a.head];
            let mut rejected = false;
            for &other in previous.iter() {
                stats.dominance_checks += 1;
                if dominates(g, &labels[other], &next, controls.max_route_arcs.is_some()) {
                    rejected = true;
                    break;
                }
            }
            if rejected {
                stats.dominated += 1;
                continue;
            }
            previous.retain(|&other| {
                stats.dominance_checks += 1;
                if dominates(g, &next, &labels[other], controls.max_route_arcs.is_some()) {
                    labels[other].active = false;
                    stats.dominated += 1;
                    false
                } else {
                    true
                }
            });
            let new_id = labels.len();
            labels.push(next);
            previous.push(new_id);
            queue.push_back(new_id);
        }
    }
    Ok(PricingResult {
        columns: pool,
        minimum_reduced_cost: minimum,
        exact: true,
        statistics: stats,
    })
}

/// Native bucket pricing. Equality resources with a non-fixed sink use the
/// supported forward search, since a backward equality root is undefined.
pub fn price<const D: usize>(g: &Graph<D>, controls: &Controls) -> Result<PricingResult> {
    g.validate()?;
    let bidirectional = !g.resources.iter().enumerate().any(|(r, k)| {
        *k == ResourceKind::Equality
            && g.vertices[g.sink].bounds[r].lower != g.vertices[g.sink].bounds[r].upper
    });
    let phase = crate::pricing::phases::Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    let mut engine = engine::LabelingEngine::prepare(
        g,
        engine::EngineControls {
            bidirectional,
            ..Default::default()
        },
        phase,
    )?;
    engine.solve(g, controls)
}
