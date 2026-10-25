//! Fractional candidates of the native routing branch families.
use super::*;
pub(crate) mod resource_groups;

#[derive(Clone, Debug)]
pub struct BranchCandidate {
    pub down: BranchConstraint,
    pub up: BranchConstraint,
    pub value: f64,
}
fn fractional(x: f64) -> bool {
    (x - x.round()).abs() > 1e-6
}
fn order(candidates: &mut [BranchCandidate]) {
    candidates.sort_by(|a, b| {
        ((a.value - a.value.floor()) - 0.5)
            .abs()
            .total_cmp(&((b.value - b.value.floor()) - 0.5).abs())
    });
}
/// Packing sets in RootProblem::set_partitioning are its customer vertices.
/// Assignment to this sole pricing network is integral once cover rows hold.
pub fn candidates<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
) -> Vec<BranchCandidate> {
    candidates_with_history(problem, result, options, &[])
}
pub fn candidates_with_history<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
    history: &[BranchConstraint],
) -> Vec<BranchCandidate> {
    fresh_candidates(problem, result, options, history, usize::MAX)
}
pub(super) fn legacy_candidates_with_history<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
    history: &[BranchConstraint],
) -> Vec<BranchCandidate> {
    generate(problem, result, options, history, false, true)
}
pub(super) fn fresh_candidates<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
    history: &[BranchConstraint],
    quota: usize,
) -> Vec<BranchCandidate> {
    fresh_candidates_with_budget(problem, result, options, history, quota, None)
}
pub(super) fn fresh_candidates_with_budget<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
    history: &[BranchConstraint],
    quota: usize,
    budget: Option<&mut super::strong::CandidateBudget>,
) -> Vec<BranchCandidate> {
    if options.priorities.validate().is_err() {
        return vec![];
    }
    super::history::fresh_batches_with_budget(
        quota,
        &super::history::registered_priorities(options),
        budget,
        |priority| {
            let mut current = options.clone();
            current.number_of_vehicles &= options.priorities.number_of_vehicles == priority;
            current.accumulated_resource &= options.priorities.accumulated_resource == priority;
            current.packing_set_assignment &= options.priorities.packing_set_assignment == priority;
            current.ryan_foster &= options.priorities.ryan_foster == priority;
            if options.priorities.aggregate_flow != priority {
                current.aggregates.clear();
                current.edge_flow = false;
            }
            if options.priorities.clusters != priority {
                current.clusters = None;
            }
            generate(
                problem,
                result,
                &current,
                history,
                true,
                options.priorities.arc_flow == priority,
            )
        },
    )
    .0
}
fn generate<const D: usize>(
    problem: &RootProblem<D>,
    result: &root::RootOutput<D>,
    options: &BranchingOptions,
    history: &[BranchConstraint],
    all: bool,
    arcs: bool,
) -> Vec<BranchCandidate> {
    let routes: Vec<_> = result
        .route_values()
        .into_iter()
        .filter(|(_, value)| *value > 1e-6)
        .collect();
    let graph_id = problem.graph.graph_id.max(0);
    let mut vehicles = vec![];
    if options.number_of_vehicles {
        let value = routes
            .iter()
            .filter(|(r, _)| r.nodes.len() > 2)
            .map(|(_, x)| x)
            .sum::<f64>();
        if fractional(value) {
            vehicles.push(BranchCandidate {
                down: BranchConstraint::NumberOfVehicles {
                    graph_id,
                    sense: RowSense::LessEqual,
                    rhs: value.floor(),
                },
                up: BranchConstraint::NumberOfVehicles {
                    graph_id,
                    sense: RowSense::GreaterEqual,
                    rhs: value.ceil(),
                },
                value,
            });
            if !all {
                return vehicles;
            }
        }
    }
    let mut candidates = if let Some(config) = &options.clusters {
        // Tree solve validates and prepares this partition once. Direct callers
        // can call clusters::partition first to diagnose invalid partitions.
        let groups = super::clusters::partition(problem, config).unwrap_or_default();
        super::clusters::candidates(problem, result, &groups)
    } else {
        vec![]
    };
    if options.accumulated_resource {
        candidates.extend(resource_groups::candidates(problem, &routes, history));
    }
    let aggregate_candidates = !candidates.is_empty();
    candidates.extend(vehicles);

    if options.packing_set_assignment {
        for &vertex in &problem.customers {
            let value = routes
                .iter()
                .map(|(r, x)| r.nodes.iter().filter(|&&v| v == vertex as i32).count() as f64 * x)
                .sum::<f64>();
            if value > 1e-6 && value < 1. - 1e-6 {
                candidates.push(BranchCandidate {
                    down: BranchConstraint::PackingSetAssignment {
                        vertex,
                        graph_id,
                        assigned: false,
                    },
                    up: BranchConstraint::PackingSetAssignment {
                        vertex,
                        graph_id,
                        assigned: true,
                    },
                    value,
                });
            }
        }
        if !all && !candidates.is_empty() && !aggregate_candidates {
            order(&mut candidates);
            return candidates;
        }
    }
    let binary_resources: usize = history
        .iter()
        .filter_map(|b| {
            if let BranchConstraint::RyanFoster(c) = b {
                Some(if c.together { 3 } else { 1 })
            } else {
                None
            }
        })
        .sum();
    if options.ryan_foster && binary_resources + 3 <= 512 {
        let supports: Vec<_> = routes
            .iter()
            .map(|(r, x)| (r.nodes.iter().copied().collect::<HashSet<_>>(), *x))
            .collect();
        for (i, &first) in problem.customers.iter().enumerate() {
            for &second in &problem.customers[i + 1..] {
                if history.iter().any(|b| {
                    matches!(b,BranchConstraint::RyanFoster(c)
                    if (c.first==first && c.second==second) || (c.first==second && c.second==first))
                }) {
                    continue;
                }
                let value = supports
                    .iter()
                    .filter(|(s, _)| s.contains(&(first as i32)) && s.contains(&(second as i32)))
                    .map(|(_, x)| x)
                    .sum::<f64>();
                if value > 1e-6 && value < 1. - 1e-6 {
                    let c = crate::pricing::restrictions::RyanFosterConstraint {
                        first,
                        second,
                        together: false,
                    };
                    candidates.push(BranchCandidate {
                        down: BranchConstraint::RyanFoster(c),
                        up: BranchConstraint::RyanFoster(
                            crate::pricing::restrictions::RyanFosterConstraint {
                                together: true,
                                ..c
                            },
                        ),
                        value,
                    });
                }
            }
        }
        if !all && !candidates.is_empty() && !aggregate_candidates {
            order(&mut candidates);
            return candidates;
        }
    }
    let mut flows = BTreeMap::<usize, f64>::new();
    for (r, value) in &routes {
        for &id in &r.arcs {
            *flows.entry(problem.arc_position(id).unwrap()).or_default() += *value;
        }
    }
    if options.aggregate_flow() {
        let registered = options
            .aggregates
            .iter()
            .filter_map(|v| Some((v.name().to_owned(), v.positions(problem)?)));
        let edges = options
            .edge_flow
            .then(|| crate::master::branching::aggregate_edges(problem))
            .unwrap_or_default();
        for (name, members) in registered.chain(edges) {
            let value = members
                .iter()
                .map(|&(arc, c)| c * flows.get(&arc).copied().unwrap_or(0.))
                .sum::<f64>();
            if fractional(value) {
                let branch = |sense, rhs| BranchConstraint::AggregateFlow {
                    name: name.clone(),
                    arcs: members.clone(),
                    sense,
                    rhs,
                };
                candidates.push(BranchCandidate {
                    down: branch(RowSense::LessEqual, value.floor()),
                    up: branch(RowSense::GreaterEqual, value.ceil()),
                    value,
                });
            }
        }
        if !all && !candidates.is_empty() && !aggregate_candidates {
            order(&mut candidates);
            return candidates;
        }
    }
    if !arcs {
        order(&mut candidates);
        return candidates;
    }
    for (arc, value) in flows {
        if fractional(value) {
            candidates.push(BranchCandidate {
                down: BranchConstraint::ArcFlow {
                    arc,
                    sense: RowSense::LessEqual,
                    rhs: value.floor(),
                },
                up: BranchConstraint::ArcFlow {
                    arc,
                    sense: RowSense::GreaterEqual,
                    rhs: value.ceil(),
                },
                value,
            });
        }
    }
    order(&mut candidates);
    candidates
}
