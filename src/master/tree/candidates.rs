//! Fractional candidates of the native routing branch families.
use super::*;

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
    let routes: Vec<_> = result
        .route_values()
        .into_iter()
        .filter(|(_, value)| *value > 1e-6)
        .collect();
    let graph_id = problem.graph.graph_id.max(0);
    if options.number_of_vehicles {
        let value = routes
            .iter()
            .filter(|(r, _)| r.nodes.len() > 2)
            .map(|(_, x)| x)
            .sum::<f64>();
        if fractional(value) {
            return vec![BranchCandidate {
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
            }];
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
        candidates.extend(resource_candidates(problem, &routes, history));
    }
    let aggregate_candidates = !candidates.is_empty();

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
        if !candidates.is_empty() && !aggregate_candidates {
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
        if !candidates.is_empty() && !aggregate_candidates {
            order(&mut candidates);
            return candidates;
        }
    }
    let mut flows = BTreeMap::<usize, f64>::new();
    for (r, value) in routes {
        for &id in &r.arcs {
            *flows.entry(problem.arc_position(id).unwrap()).or_default() += value;
        }
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

/// Split the observed head-resource schedules of an arc with unit LP flow.
/// Fractional arc flows remain handled by the ordinary arc-flow fallback.
fn resource_candidates<const D: usize>(
    problem: &RootProblem<D>,
    routes: &[(&crate::Route, f64)],
    history: &[BranchConstraint],
) -> Vec<BranchCandidate> {
    use crate::pricing::restrictions::AccumulatedResourceBranch;
    let mut observations = BTreeMap::<(usize, usize), Vec<(f64, f64)>>::new();
    for &(route, value) in routes {
        for (i, &id) in route.arcs.iter().enumerate() {
            let Some(arc) = problem.arc_position(id) else {
                continue;
            };
            for resource in 0..problem.graph.resources.len() {
                if let Some(&q) = route
                    .resource_profile
                    .get(i + 1)
                    .and_then(|q| q.get(resource))
                {
                    if q.is_finite() {
                        observations
                            .entry((arc, resource))
                            .or_default()
                            .push((q, value));
                    }
                }
            }
        }
    }
    let mut candidates = vec![];
    for ((arc, resource), mut samples) in observations {
        if (samples.iter().map(|x| x.1).sum::<f64>() - 1.).abs() > 1e-6 {
            continue;
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (low, high) = (samples.first().unwrap().0, samples.last().unwrap().0);
        let threshold = low * 0.5 + high * 0.5;
        if threshold <= low || threshold >= high {
            continue;
        }
        let id = problem.graph.arcs[arc].id;
        if history.iter().any(|c| {
            matches!(c, BranchConstraint::AccumulatedResource(b)
            if b.arc == id && b.resource == resource && b.threshold == threshold)
        }) {
            continue;
        }
        let up = AccumulatedResourceBranch {
            arc: id,
            resource,
            threshold,
            upper: true,
        };
        let down = AccumulatedResourceBranch { upper: false, ..up };
        if !matches!(up.window(&problem.graph), Ok(Some(_)))
            || !matches!(down.window(&problem.graph), Ok(Some(_)))
        {
            continue;
        }
        let value: f64 = samples
            .iter()
            .filter(|x| x.0 > threshold)
            .map(|x| x.1)
            .sum();
        if value > 1e-6 && value < 1. - 1e-6 {
            candidates.push(BranchCandidate {
                down: BranchConstraint::AccumulatedResource(down),
                up: BranchConstraint::AccumulatedResource(up),
                value,
            });
        }
    }
    candidates
}
