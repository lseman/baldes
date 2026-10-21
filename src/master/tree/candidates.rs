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
        // Encode the intended midpoint in the stored-threshold convention.
        // Check the decoded boundary too: adding/subtracting the offset can
        // round to an observed endpoint for very narrow resource intervals.
        let threshold = low * 0.5 + high * 0.5 + AccumulatedResourceBranch::THRESHOLD_OFFSET;
        let up = AccumulatedResourceBranch {
            arc: problem.graph.arcs[arc].id,
            resource,
            threshold,
            upper: true,
        };
        let boundary = up.boundary();
        if !threshold.is_finite() || boundary <= low || boundary >= high {
            continue;
        }
        if history.iter().any(|c| {
            matches!(c, BranchConstraint::AccumulatedResource(b)
            if b.arc == up.arc && b.resource == resource && b.boundary() == boundary)
        }) {
            continue;
        }
        let down = AccumulatedResourceBranch { upper: false, ..up };
        if !matches!(up.window(&problem.graph), Ok(Some(_)))
            || !matches!(down.window(&problem.graph), Ok(Some(_)))
        {
            continue;
        }
        let value: f64 = samples.iter().filter(|x| x.0 > boundary).map(|x| x.1).sum();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::{restrictions::AccumulatedResourceBranch, Solver};

    #[test]
    fn resource_midpoints_decode_inside_the_observed_interval() {
        let mut s = Solver::<5>::new(3, 5).unwrap();
        for v in 0..3 {
            for r in 0..5 {
                s.graph.set_bounds(v, r, 0., 3.).unwrap();
            }
        }
        let arc = s.graph.add_arc(0, 1, 0.).unwrap();
        let end = s.graph.add_arc(1, 2, 0.).unwrap();
        let problem = RootProblem::set_partitioning(s.graph, 1.).unwrap();
        let route = |q| {
            let mut head = vec![0.; 5];
            head[4] = q;
            Route {
                nodes: vec![0, 1, 2],
                arcs: vec![arc, end],
                cost: 0.,
                original_cost: 0.,
                resources: vec![3.; 5],
                resource_profile: vec![vec![0.; 5], head, vec![3.; 5]],
            }
        };
        let low = route(1. - 0.5e-6);
        let high = route(1. + 0.5e-6);
        let observations = [(&low, 0.25), (&high, 0.75)];
        let candidates = resource_candidates(&problem, &observations, &[]);
        assert_eq!(candidates.len(), 1);
        let BranchConstraint::AccumulatedResource(up) = candidates[0].up else {
            panic!()
        };
        let BranchConstraint::AccumulatedResource(down) = candidates[0].down else {
            panic!()
        };
        assert_eq!(up.resource, 4);
        assert_eq!(
            up.threshold,
            1. + AccumulatedResourceBranch::THRESHOLD_OFFSET
        );
        assert_eq!(up.boundary(), down.boundary());
        assert!(!up.satisfied_by(low.resource_profile[1][4]));
        assert!(!down.satisfied_by(high.resource_profile[1][4]));
        assert_eq!(candidates[0].value, 0.75);
        for previous in [candidates[0].down.clone(), candidates[0].up.clone()] {
            assert!(resource_candidates(&problem, &observations, &[previous]).is_empty());
        }
        // At this scale the offset loses the midpoint during round-trip
        // conversion. Skip a degenerate split instead of proposing it forever.
        let low = route(0.);
        let high = route(1e-30);
        assert!(resource_candidates(&problem, &[(&low, 0.5), (&high, 0.5)], &[]).is_empty());
    }
}
