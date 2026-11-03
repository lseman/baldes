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
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            None,
        )
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
        .prepare(graphs(), None, None)
        .is_err());
    let mut labelled = RyanFosterBranching::new(2.);
    labelled.sets = vec![
        NetworkPackingAssignment::new(0, vec![(1, 0), (2, 1)]),
        NetworkPackingAssignment::new(1, vec![(1, 0)]),
    ];
    labelled.prepare(graphs(), None, None).unwrap();
    // Set 1 is absent from network 1 while set 0 is present there.
    assert!(labelled.members(0, 1).is_none());
    labelled.sets = vec![NetworkPackingAssignment::new(0, vec![(0, 0)])];
    assert!(labelled.prepare(graphs(), None, None).is_err());
    assert!(RyanFosterBranching::new(f64::NAN)
        .prepare(graphs().take(1), None, None)
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

#[test]
fn explicit_incidence_candidates_install_arc_sensitive_children() {
    let (problem, config) = super::super::separation::mixed_arc_problem_and_config();
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            Some(&config),
        )
        .unwrap();
    let mut root = solve_with_master_variables(problem, vec![], options()).unwrap();
    let mut route = root.routes[0].route.clone();
    route.arcs = vec![0, 3, 15];
    root.routes = vec![NetworkRoute { network: 7, route }];
    root.values = vec![0.5];
    let proposals = branching.proposals(&root).unwrap();
    let proposal = proposals.iter().find(|p| p.name == "PS_0_2").unwrap();
    let Restriction::ArcSetRyanFoster { members } = &proposal.restriction else {
        panic!("expected arc-set branch")
    };
    assert_eq!(members.len(), 2);
    for together in [false, true] {
        let child = root
            .snapshot
            .with_arc_set_branch(members, together)
            .unwrap();
        let row = child.state.shared_rows.len() - 1;
        for oracle in &child.state.oracles {
            let constraint = &oracle.arc_set_branch_rows.last().unwrap().1;
            for original in root
                .snapshot
                .state
                .routes
                .iter()
                .filter(|r| r.network == oracle.network.id)
            {
                let entries = super::entries(oracle, &original.route).unwrap();
                let coefficient = entries.iter().any(|&(r, _)| r == row);
                assert_eq!(
                    coefficient,
                    !constraint.satisfied_by(original.route.arcs.iter().copied())
                );
            }
        }
        let solved = child.solve(options()).unwrap();
        for (route, &value) in solved.routes.iter().zip(&solved.values) {
            if value > 1e-8 {
                let oracle = solved
                    .snapshot
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network)
                    .unwrap();
                let constraint = &oracle.arc_set_branch_rows.last().unwrap().1;
                assert!(constraint.satisfied_by(route.route.arcs.iter().copied()));
            }
        }
        assert!(root
            .snapshot
            .state
            .oracles
            .iter()
            .all(|o| o.arc_set_branch_rows.is_empty()));
    }
}

#[test]
fn explicit_incidence_candidate_values_match_native_capture() {
    use crate::cuts::separation::{ArcData, GraphData, Parameters, VertexData};
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/ryan_foster_candidate_values_native.json"
    ))
    .unwrap();
    let case: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/ryan_foster_candidate_model_native.json"
    ))
    .unwrap();
    let problem = crate::master::model::format::Model::from_json(&case["model"].to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let graphs = problem
        .networks
        .iter()
        .map(|network| {
            let graph = &network.graph;
            let vertices = (0..10)
                .map(|id| {
                    let sets = if id == 1 {
                        vec![1]
                    } else if (3..=8).contains(&id) {
                        vec![id - 1]
                    } else {
                        vec![]
                    };
                    VertexData {
                        id,
                        elem_sets: sets.clone(),
                        packing_sets: sets.clone(),
                        covering_sets: sets,
                    }
                })
                .collect();
            let arcs = graph
                .arcs
                .iter()
                .map(|arc| {
                    let sets = if arc.head == 1 && arc.tail != 4 && arc.tail != 5 {
                        vec![0]
                    } else {
                        vec![]
                    };
                    ArcData {
                        id: arc.id,
                        tail: arc.tail as i32,
                        head: arc.head as i32,
                        elem_sets: sets.clone(),
                        packing_sets: sets.clone(),
                        covering_sets: sets,
                    }
                })
                .collect();
            GraphData {
                id: network.id,
                source: 0,
                sink: 9,
                elem_set_count: 8,
                packing_set_count: 8,
                covering_set_count: 8,
                vertices,
                arcs,
                ..Default::default()
            }
        })
        .collect();
    let config = RankOneIncidenceConfig {
        rows: (0..8).collect(),
        graphs,
        parameters: Parameters::default(),
    };
    let mut branching = RyanFosterBranching::new(2.);
    branching
        .prepare(
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            None,
            Some(&config),
        )
        .unwrap();
    let mut output = solve_with_master_variables(problem, vec![], options()).unwrap();
    let template = output.routes[0].route.clone();
    assert!(output.snapshot.state.fixed_routes.is_empty());
    assert_eq!(capture["exit_code"], 0);
    for hit in capture["hits"].as_array().unwrap() {
        output.routes.clear();
        output.values.clear();
        for captured in hit["routes"].as_array().unwrap() {
            let mut route = template.clone();
            route.arcs = captured["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            output.routes.push(NetworkRoute {
                network: captured["graph"].as_i64().unwrap() as i32,
                route,
            });
            output.values.push(captured["quantity"].as_f64().unwrap());
        }
        let name = format!("PS_{}_{}", hit["pair"][0], hit["pair"][1]);
        let expected = hit["value"].as_f64().unwrap();
        let proposals = branching.proposals(&output).unwrap();
        let actual = proposals
            .iter()
            .find(|p| p.name == name)
            .map_or(0., |p| p.value);
        assert!(
            (actual - expected).abs() < 1e-9,
            "{name}: {actual} != {expected}"
        );
    }
}
