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
pub fn candidates(
    problem: &RootProblem,
    result: &root::RootOutput,
    options: &BranchingOptions,
) -> Vec<BranchCandidate> {
    candidates_with_history(problem, result, options, &[])
}
pub fn candidates_with_history(
    problem: &RootProblem,
    result: &root::RootOutput,
    options: &BranchingOptions,
    history: &[BranchConstraint],
) -> Vec<BranchCandidate> {
    let routes: Vec<_> = result.route_values().into_iter()
        .filter(|(_, value)| *value > 1e-6).collect();
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
    let mut candidates = vec![];
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
        if !candidates.is_empty() {
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
                    let c = crate::pricing::branching::RyanFosterConstraint {
                        first,
                        second,
                        together: false,
                    };
                    candidates.push(BranchCandidate {
                        down: BranchConstraint::RyanFoster(c),
                        up: BranchConstraint::RyanFoster(
                            crate::pricing::branching::RyanFosterConstraint {
                                together: true,
                                ..c
                            },
                        ),
                        value,
                    });
                }
            }
        }
        if !candidates.is_empty() {
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
