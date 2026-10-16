//! Forward no-cut labeling with immutable predecessor identities.
pub mod engine;
pub mod pool;
pub mod storage;
use crate::graph::ng::PreparedNg;
use crate::graph::{Arc, Graph, ResourceKind};
use crate::{incomplete, invalid, Result};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const EPSILON: f64 = 1e-6;
pub const NO_ROUTE_COST: f64 = 1e12;
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
            max_labels: 1_000_000,
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
}
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
#[derive(Debug)]
pub struct PricingResult {
    pub columns: Vec<Route>,
    pub minimum_reduced_cost: f64,
    pub exact: bool,
    pub statistics: Statistics,
}
#[derive(Clone, Debug)]
struct Label {
    vertex: usize,
    cost: f64,
    original_cost: f64,
    resources: [f64; 2],
    ng: u16,
    parent: Option<usize>,
    arc: Option<usize>,
    hops: usize,
    active: bool,
}

/// StdResources transitions: disposable resources may wait at a lower bound;
/// equality resources cannot. Backward propagation clips before subtraction.
pub fn extend_resources(g: &Graph, a: &Arc, mut q: [f64; 2], forward: bool) -> Option<[f64; 2]> {
    for (r, kind) in g.resources.iter().enumerate() {
        let mut head = g.vertices[a.head].bounds[r];
        if let Some(window) = a.bounds[r] {
            head.lower = head.lower.max(window.lower);
            head.upper = head.upper.min(window.upper);
        }
        let mut tail = g.vertices[a.tail].bounds[r];
        if a.tail == g.source {
            if let Some(initial) = g.initial[r] {
                tail.lower = tail.lower.max(initial);
            }
        }
        if forward {
            q[r] += a.consumption[r];
            if q[r] < head.lower {
                if *kind == ResourceKind::Equality {
                    return None;
                }
                q[r] = head.lower;
            }
            if q[r] > head.upper {
                return None;
            }
        } else {
            if q[r] > head.upper {
                if *kind == ResourceKind::Equality {
                    return None;
                }
                q[r] = head.upper;
            }
            if q[r] < head.lower {
                return None;
            }
            q[r] -= a.consumption[r];
            if q[r] > tail.upper {
                if *kind == ResourceKind::Equality {
                    return None;
                }
                q[r] = tail.upper;
            }
            if q[r] < tail.lower {
                return None;
            }
        }
        if !q[r].is_finite() {
            return None;
        }
    }
    Some(q)
}
fn dominates(g: &Graph, a: &Label, b: &Label, hop_limit: bool) -> bool {
    if a.vertex != b.vertex
        || a.cost - b.cost >= EPSILON
        || a.ng & !b.ng != 0
        || (hop_limit && a.hops > b.hops)
    {
        return false;
    }
    g.resources.iter().enumerate().all(|(r, k)| match k {
        ResourceKind::Disposable => a.resources[r] <= b.resources[r] + EPSILON,
        ResourceKind::Equality => {
            a.resources[r] <= b.resources[r] + EPSILON && b.resources[r] - EPSILON <= a.resources[r]
        }
    })
}
fn reconstruct(g: &Graph, labels: &[Label], terminal: &Label) -> Route {
    let mut arcs = vec![];
    let mut next = Some(terminal);
    while let Some(l) = next {
        if let Some(a) = l.arc {
            arcs.push(a);
        }
        next = l.parent.map(|id| &labels[id]);
    }
    arcs.reverse();
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
    }
}

/// Complete a label-correcting search or return an error. Budget/time exhaustion
/// never returns a partial pool with an exactness certificate.
pub fn reference_price(g: &Graph, controls: &Controls) -> Result<PricingResult> {
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
    let mut q = [0.; 2];
    for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
        *value = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
        let w = g.vertices[g.source].bounds[r];
        if *value < w.lower || *value > w.upper {
            return Err(invalid("initial resource outside source window"));
        }
    }
    let root = Label {
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
            let Some(resources) = extend_resources(g, a, from.resources, true) else {
                continue;
            };
            let Some(memory) = ng.forward[index].extend(from.ng, ng.vertex_bits[a.head]) else {
                continue;
            };
            let next = Label {
                vertex: a.head,
                cost: from.cost + a.reduced_cost,
                original_cost: from.original_cost + a.cost,
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
pub fn price(g: &Graph, controls: &Controls) -> Result<PricingResult> {
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
