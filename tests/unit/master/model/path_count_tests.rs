use super::*;

fn problem() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}

fn root() -> Output<2> {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    solve::solve(problem(), options).unwrap()
}

fn registrations() -> Vec<PathsPerNetworkBranching> {
    vec![PathsPerNetworkBranching {
        network: 42,
        priority: -2.5,
    }]
}

#[test]
fn explicit_path_count_uses_family_precision_independent_of_tree_tolerance() {
    let mut output = root();
    let tag = output.routes.iter().position(|r| r.network == 42).unwrap();
    for (value, accepted) in [
        (0., false),
        (0.5e-8, false),
        (1e-8, true),
        (0.005, true),
        (0.5, true),
        (1. - 1e-8, true),
        (1. - 0.5e-8, false),
        (1., false),
    ] {
        output.values.fill(0.);
        output.values[tag] = value;
        let candidates = proposals(&output, &registrations(), 0.1).unwrap();
        assert_eq!(!candidates.is_empty(), accepted, "value={value}");
        if accepted {
            let candidate = &candidates[0];
            assert_eq!(candidate.name, "paths:42");
            assert_eq!(candidate.priority, -2.5);
            assert!(candidate.history_eligible);
        }
    }
}

#[test]
fn explicit_path_count_keeps_fixed_route_copies_in_original_activity() {
    let mut output = root();
    let tag = output.routes.iter().position(|r| r.network == 42).unwrap();
    output.values.fill(0.);
    output.values[tag] = 0.5;
    output.snapshot.state.fixed_routes.push(FixedRoute {
        route: output.routes[tag].clone(),
        quantity: 2.,
    });
    let candidates = proposals(&output, &registrations(), 1e-6).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].value, 2.5);
}

#[test]
fn registered_path_count_child_rows_restrict_repriced_routes() {
    use crate::master::preprocessing::Bounds;

    let output = root();
    let candidate = proposals(&output, &registrations(), 1e-6)
        .unwrap()
        .pop()
        .unwrap();
    assert!((candidate.value - 0.5).abs() < 1e-8);
    for bounds in [
        Bounds {
            lower: 0.,
            upper: candidate.value.floor(),
        },
        Bounds {
            lower: candidate.value.ceil(),
            upper: f64::INFINITY,
        },
    ] {
        let child = output.snapshot.with_multiplicity(42, bounds).unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        let solved = child.solve(options).unwrap();
        assert_eq!(solved.master.status, cg::Status::Optimal);
        let count = solved
            .routes
            .iter()
            .zip(&solved.values)
            .filter(|(route, _)| route.network == 42)
            .map(|(_, &quantity)| quantity)
            .sum::<f64>();
        assert!(count >= bounds.lower - 1e-8);
        assert!(count <= bounds.upper + 1e-8);
    }
}

#[test]
fn path_count_registration_rejects_unknown_duplicate_and_ambiguous_domains() {
    let input = problem();
    let graphs = || input.networks.iter().map(|n| (n.id, &n.graph));
    validate(&registrations(), graphs()).unwrap();
    assert!(validate(
        &[PathsPerNetworkBranching {
            network: 999,
            priority: 0.,
        }],
        graphs()
    )
    .is_err());
    let mut duplicate = registrations();
    duplicate.extend(registrations());
    assert!(validate(&duplicate, graphs()).is_err());
    let mut invalid_priority = registrations();
    invalid_priority[0].priority = f64::NAN;
    assert!(validate(&invalid_priority, graphs()).is_err());
    let original = input.networks.iter().find(|n| n.id == 42).unwrap();
    for modification in 0..4 {
        let mut graph = original.graph.clone();
        match modification {
            0 => graph.arcs[0].head = graph.sink,
            1 => graph.arcs[0].head = graph.source,
            2 => graph.arcs[0].tail = graph.sink,
            _ => graph.sink = graph.source,
        }
        assert!(validate(&registrations(), std::iter::once((42, &graph))).is_err());
    }
}
