use super::*;
use crate::inputs::{Instance, Rounding};

fn rc101(customers: usize) -> Problem {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    Problem::set_partitioning(graph, customers as f64).unwrap()
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options
}

fn prepared(problem: &Problem) -> RyanFosterBranching {
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(problem.networks.iter().map(|n| (n.id, &n.graph)), None)
        .unwrap();
    branching
}

#[test]
fn candidates_are_fractional_cooccurrences_and_children_obey_the_pair() {
    let problem = rc101(25);
    let branching = prepared(&problem);
    let root = solve_with_master_variables(problem, vec![], options()).unwrap();
    let proposals = branching.proposals(&root).unwrap();
    assert!(!proposals.is_empty(), "the RC101 root is fractional");
    for p in &proposals {
        assert!(p.value > 1e-6 && p.value < 0.999999, "{}", p.value);
        assert!(p.name.starts_with("PS_"));
        assert_eq!(
            p.origin,
            Some(super::model_strong::ProducerFamily::PackingRyanFoster)
        );
    }
    let best = proposals
        .iter()
        .min_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()))
        .unwrap();
    let Restriction::RyanFoster { members } = &best.restriction else {
        unreachable!()
    };
    let member = members[0];
    for together in [false, true] {
        let child = root.snapshot.with_ryan_foster(members, together).unwrap();
        let solved = child.solve(options()).unwrap();
        assert!(solved.master.bound >= root.master.bound - 1e-6);
        let constraint = RyanFosterConstraint {
            first: member.first,
            second: member.second,
            together,
        };
        let oracle = &solved.snapshot.state.oracles[0];
        for (route, &value) in solved.routes.iter().zip(&solved.values) {
            if value > 1e-9 {
                let vertices: Vec<_> = path_vertices(oracle, &route.route).unwrap().collect();
                assert!(
                    constraint.satisfied_by(vertices.iter().copied()),
                    "{together} {vertices:?}"
                );
            }
        }
    }
    // The parent snapshot is unchanged by forming children.
    assert!(root.snapshot.state.oracles[0].ryan_foster_rows.is_empty());
}

#[test]
fn several_networks_require_packing_sets_and_customer_vertices() {
    let mut problem = rc101(5);
    let mut second = problem.networks[0].clone();
    second.id = 1;
    problem.networks.push(second);
    let graphs = || problem.networks.iter().map(|n| (n.id, &n.graph));
    assert!(RyanFosterBranching::new(2.)
        .prepare(graphs(), None)
        .is_err());
    let mut labelled = RyanFosterBranching::new(2.);
    labelled.sets = vec![
        NetworkPackingAssignment::new(0, vec![(1, 0), (2, 1)]),
        NetworkPackingAssignment::new(1, vec![(1, 0)]),
    ];
    labelled.prepare(graphs(), None).unwrap();
    // Set 1 is absent from network 1 while set 0 is present there.
    assert!(labelled.members(0, 1).is_none());
    labelled.sets = vec![NetworkPackingAssignment::new(0, vec![(0, 0)])];
    assert!(labelled.prepare(graphs(), None).is_err());
    assert!(RyanFosterBranching::new(f64::NAN)
        .prepare(graphs().take(1), None)
        .is_err());
}

#[test]
fn ryan_foster_tree_reaches_the_default_branching_optimum() {
    let problem = rc101(25);
    let solve = |ryan_foster: Option<RyanFosterBranching>| {
        let mut configured = ModelTreeOptions::default();
        configured.branching_registry.mapped_variables = problem.arc_flow_variables(1.);
        configured.branching_registry.ryan_foster = ryan_foster;
        configured.node_limit = 2000;
        crate::master::tree::solve(problem.clone(), configured).unwrap()
    };
    let default = solve(None);
    let paired = solve(Some(RyanFosterBranching::new(2.)));
    assert_eq!(default.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(paired.status, crate::master::tree::TreeStatus::Optimal);
    assert!(paired.processed_nodes > 1);
    assert!(
        (default.incumbent - paired.incumbent).abs() < 1e-6,
        "{} {}",
        default.incumbent,
        paired.incumbent
    );
}
