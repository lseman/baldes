use super::*;
use crate::graph::coupled::{
    plugin::{CoupledPlugin, PluginResource, PluginState},
    CoupledResource,
};
use crate::master::preprocessing::Bounds;

/// The physical relaxation can visit three customers, but original routes may
/// serve at most two. Like robust PD pairing, this plugin certifies individual
/// routes without supplying a whole-graph phase-I pricing certificate.
#[derive(Debug)]
struct AtMostTwo {
    arcs: Vec<(usize, usize)>,
}

impl CoupledPlugin for AtMostTwo {
    fn validate(&self, arcs: usize) -> Result<()> {
        assert_eq!(arcs, self.arcs.len());
        Ok(())
    }

    fn root(&self, _forward: bool) -> PluginState {
        PluginState::new(0u8)
    }

    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<PluginState>> {
        let (tail, head) = self.arcs[arc];
        let v = if forward { head } else { tail };
        let mut visited = *state.get::<u8>().unwrap();
        if (1..=3).contains(&v) {
            let bit = 1 << (v - 1);
            if visited & bit != 0 {
                return Ok(None);
            }
            visited |= bit;
        }
        Ok((visited.count_ones() <= 2).then(|| PluginState::new(visited)))
    }

    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        let mut state = state.clone();
        for arc in suffix {
            let Some(next) = self.extend(&state, arc, true)? else {
                return Ok(false);
            };
            state = next;
        }
        Ok(true)
    }

    fn certify(&self, arcs: &[usize], work: &mut usize, deadline: Option<Instant>) -> Option<bool> {
        *work = work.checked_sub(arcs.len() + 1)?;
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        let mut node = 0;
        let mut visited = 0u8;
        for &arc in arcs {
            let &(tail, head) = self.arcs.get(arc)?;
            if tail != node || node == 4 {
                return Some(false);
            }
            node = head;
            if (1..=3).contains(&head) {
                let bit = 1 << (head - 1);
                if visited & bit != 0 {
                    return Some(false);
                }
                visited |= bit;
            }
        }
        Some(node == 4 && (1..=2).contains(&visited.count_ones()))
    }
}

fn problem(max_routes: f64) -> Problem<2> {
    let mut graph = Graph::<2>::new(5, 1).unwrap();
    let mut arcs = vec![];
    let mut entries = vec![];
    for tail in 0..4 {
        for head in 1..=4 {
            if tail == head || (tail == 0 && head == 4) {
                continue;
            }
            graph
                .add_arc(tail, head, if tail == 0 { 1. } else { 0. })
                .unwrap();
            arcs.push((tail, head));
            entries.push(if head < 4 {
                vec![(head - 1, 1.)]
            } else {
                vec![]
            });
        }
    }
    graph
        .set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(AtMostTwo {
            arcs,
        }))])
        .unwrap();
    Problem {
        rows: (0..3)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![Network {
            id: 0,
            graph,
            multiplicity: Window::new(0., max_routes).unwrap(),
            arc_entries: entries,
            route_entries: vec![],
        }],
    }
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.artificial_cost = 100.;
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options
}

#[test]
fn stalled_child_bound_closes_tree_against_valid_incumbent() {
    let root = solve_with_master_variables(problem(3.), vec![], options()).unwrap();
    assert_eq!(root.master.status, cg::Status::Optimal);
    assert!((root.master.lp - 1.5).abs() < 1e-8);
    let lower = root
        .snapshot
        .with_multiplicity(
            0,
            Bounds {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap()
        .solve(options())
        .unwrap();
    assert_eq!(lower.master.status, cg::Status::Stalled);
    assert!(lower.master.bound > 2.);
    assert!(!lower
        .master
        .feasibility
        .iter()
        .any(|f| f.certified_infeasible));

    let result = crate::master::tree::solve(
        problem(3.),
        ModelTreeOptions {
            node: options(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, TreeStatus::Optimal, "{result:?}");
    assert!((result.incumbent - 2.).abs() < 1e-8);
    assert!((result.bound - 2.).abs() < 1e-8);
    assert_eq!(result.open_nodes, 0);
    assert_eq!(result.routes.iter().map(|r| r.quantity).sum::<f64>(), 2.);
}

#[test]
fn stalled_node_without_an_incumbent_does_not_prove_infeasibility() {
    let result = crate::master::tree::solve(
        problem(1.),
        ModelTreeOptions {
            node: options(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete, "{result:?}");
    assert_eq!(result.incumbent, f64::INFINITY);
    assert!(result.open_nodes > 0);
}

#[test]
fn uncertified_stalled_bound_does_not_close_a_certified_tree() {
    let mut options = options();
    options.cg.bound_mode = cg::BoundMode::Certified;
    let result = crate::master::tree::solve(
        problem(3.),
        ModelTreeOptions {
            node: options,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete, "{result:?}");
    assert_eq!(result.bound, f64::NEG_INFINITY);
    assert!(result.open_nodes > 0);
}
