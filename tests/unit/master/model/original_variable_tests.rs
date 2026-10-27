use super::*;

#[test]
fn original_rounding_precedes_association_and_survives_resume_without_option_reinstall() {
    let mut v = variable();
    v.cost = -0.1234567893;
    v.integer = false;
    v.arcs = vec![(0, 0.5), (2, 0.5)];
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let (problem, registry) = compile_pricing_variables(alternatives(), &[v.clone()]).unwrap();
        let mut configured = options();
        configured.cg.bound_mode = mode;
        registry.configure_pricing(&mut configured);
        let output = solve(problem, configured).unwrap();
        let raw = v.cost * 0.5;
        assert_eq!(output.master.lp, raw);
        assert!(output.master.bound <= raw);
        assert!(raw - output.master.bound < 1e-12);
        let cached = output.snapshot.state.oracles[0].solver.graph.arcs[0].reduced_cost;
        assert_eq!(cached, -0.061728395);
        assert_ne!(cached, (raw * 1e8 + 0.5).floor() / 1e8);
        let resumed = output.snapshot.solve(options()).unwrap();
        assert_eq!(resumed.master.lp, raw);
        assert_eq!(
            resumed.snapshot.state.oracles[0].solver.graph.arcs[0].reduced_cost,
            cached
        );
    }
}

#[test]
fn original_cost_cache_preserves_base_costs_overlapping_associations_and_appended_rows() {
    let mut problem = alternatives();
    problem.rows.push(Row {
        sense: RowSense::Equal,
        rhs: 1.,
    });
    problem.networks[0].graph.arc_mut(0).unwrap().cost = 0.125;
    problem.networks[0].arc_entries[0] = vec![(0, 0.25)];
    let mut first = variable();
    first.integer = false;
    first.cost = -0.1234567893;
    first.entries = vec![(0, 0.125)];
    first.arcs = vec![(0, 0.5), (2, 0.5)];
    let mut second = first.clone();
    second.id = 1;
    second.cost = 0.3333333333;
    second.arcs = vec![(0, -0.25)];
    let (problem, registry) =
        compile_pricing_variables(problem, &[first.clone(), second.clone()]).unwrap();
    let mut network: OracleNetwork<2> = problem.networks.into_iter().next().unwrap().into();
    let index = network
        .graph
        .arcs
        .iter()
        .enumerate()
        .map(|(p, a)| (a.id, p))
        .collect();
    Arc::make_mut(&mut network.arc_entries)[0].push((1, 0.7));
    let round = |c: f64| (c * 1e8 + 0.5).floor() / 1e8;
    let expected = 0.125 - 0.1 * 0.25 - 0.2 * 0.7
        + round(first.cost - 0.1 * 0.125) * 0.5
        + round(second.cost - 0.1 * 0.125) * -0.25;
    let cached = registry.costs[0]
        .cached(&network, &index, &[0.1, 0.2], 0)
        .unwrap();
    assert_eq!(cached[0], expected);
    Arc::make_mut(&mut network.arc_entries)[0][0].1 = 9.;
    assert!(registry.costs[0]
        .cached(&network, &index, &[0.1, 0.2], 0)
        .is_none());
}

#[test]
fn original_cost_decomposition_copies_the_association_to_new_physical_arcs() {
    let mut v = variable();
    v.integer = false;
    v.cost = -0.1234567893;
    v.arcs = vec![(0, 0.5), (2, 0.5)];
    let (mut problem, registry) = compile_pricing_variables(alternatives(), &[v]).unwrap();
    let network = &mut problem.networks[0];
    let cost = network.graph.arcs[0].cost;
    let copy = network.graph.add_arc(0, 1, cost).unwrap();
    network.arc_entries.push(network.arc_entries[0].clone());
    let mut model = registry.costs[0].clone();
    model.copy_arcs(std::iter::once((copy, 0)));
    let network: OracleNetwork<2> = problem.networks.into_iter().next().unwrap().into();
    let index = network
        .graph
        .arcs
        .iter()
        .enumerate()
        .map(|(p, a)| (a.id, p))
        .collect();
    let cached = model.cached(&network, &index, &[], 0).unwrap();
    assert_eq!(cached[0], -0.061728395);
    assert_eq!(cached[index[&copy]], cached[0]);
}

fn alternatives() -> Problem<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    Problem {
        rows: vec![],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window {
                lower: 0.,
                upper: 1.,
            },
            arc_entries: vec![vec![]; 4],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    }
}

fn variable() -> OriginalPricingVariable {
    OriginalPricingVariable {
        accumulated_resource: None,
        network: 7,
        id: 0,
        cost: -1.,
        entries: vec![],
        arcs: vec![(0, 1.), (2, 1.)],
        integer: true,
        priority: 3.,
    }
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options
}

#[test]
fn alternative_aliases_keep_one_original_activity_and_finite_optimum() {
    let (problem, registry) = compile_pricing_variables(alternatives(), &[variable()]).unwrap();
    assert_eq!(registry.branch_variables()[0].name, "X_7_0");
    assert_eq!(
        registry.branch_variables()[0].networks[0].arcs,
        [(0, 1.), (2, 1.)]
    );
    let output = solve(problem, options()).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, -1.);
    assert_eq!(
        registry.activity(7, 0, &output).unwrap(),
        Bounds {
            lower: 1.,
            upper: 1.
        }
    );
    let mut tree = ModelTreeOptions::default();
    registry.configure_tree(&mut tree).unwrap();
    assert_eq!(tree.mapped_variables.len(), 1);
    let before = tree.mapped_variables.len();
    assert!(registry.configure_tree(&mut tree).is_err());
    assert_eq!(tree.mapped_variables.len(), before);
}

#[test]
fn setters_replace_same_variable_but_distinct_variables_and_base_rows_add() {
    let mut problem = alternatives();
    problem.rows.push(Row {
        sense: RowSense::Equal,
        rhs: 1.,
    });
    problem.networks[0].graph.arc_mut(0).unwrap().cost = 3.;
    problem.networks[0].arc_entries[0] = vec![(0, 1.)];
    let mut first = variable();
    first.arcs = vec![(0, 9.), (0, 0.5), (2, 1.), (2, 0.)];
    first.integer = false;
    first.entries = vec![(0, 2.)];
    let mut second = variable();
    second.id = 1;
    second.cost = -2.;
    second.arcs = vec![(0, 0.25)];
    second.integer = false;
    second.entries = vec![(0, 4.)];
    let (compiled, registry) = compile_pricing_variables(problem, &[second, first]).unwrap();
    assert_eq!(compiled.networks[0].graph.arcs[0].cost, 2.);
    assert_eq!(compiled.networks[0].graph.arcs[2].cost, 0.);
    assert_eq!(
        compiled.networks[0].arc_entries[0],
        [(0, 1.), (0, 1.), (0, 1.)]
    );
    assert_eq!(registry.variables()[0].arcs, [(0, 0.5)]);
    assert!(registry.branch_variables().is_empty());
    let output = solve(compiled, options()).unwrap();
    assert!((output.master.lp - 2. / 3.).abs() < 1e-8);
    let activity = registry.activity(7, 0, &output).unwrap();
    assert!(activity.lower <= 1. / 6. && activity.upper >= 1. / 6.);
}

#[test]
fn serial_half_associations_are_integer_and_signed_costs_keep_their_algebra() {
    let mut problem = alternatives();
    problem.networks[0].graph = Graph::<2>::new(3, 0).unwrap();
    problem.networks[0].graph.add_arc(0, 1, 0.).unwrap();
    problem.networks[0].graph.add_arc(1, 2, 0.).unwrap();
    problem.networks[0].arc_entries = vec![vec![]; 2];
    let mut alias = variable();
    alias.arcs = vec![(0, 0.5), (1, 0.5)];
    let (compiled, registry) =
        compile_pricing_variables(problem.clone(), &[alias.clone()]).unwrap();
    let output = solve(compiled, options()).unwrap();
    assert_eq!(output.master.lp, -1.);
    assert_eq!(
        registry.activity(7, 0, &output).unwrap(),
        Bounds {
            lower: 1.,
            upper: 1.
        }
    );
    alias.cost = 1.;
    alias.arcs = vec![(0, -0.5), (1, -0.5)];
    let (compiled, registry) = compile_pricing_variables(problem, &[alias]).unwrap();
    let output = solve(compiled, options()).unwrap();
    assert_eq!(output.master.lp, -1.);
    assert_eq!(
        registry.activity(7, 0, &output).unwrap(),
        Bounds {
            lower: -1.,
            upper: -1.
        }
    );
}

#[test]
fn invalid_identity_association_and_integer_path_claims_fail_compilation() {
    let original = alternatives();
    let mut v = variable();
    assert!(compile_pricing_variables(original.clone(), &[v.clone(), v.clone()]).is_err());
    v.network = 42;
    assert!(compile_pricing_variables(original.clone(), &[v.clone()]).is_err());
    v = variable();
    v.arcs = vec![(100, 1.)];
    assert!(compile_pricing_variables(original.clone(), &[v.clone()]).is_err());
    v.arcs = vec![(0, 0.5), (2, 0.5)];
    assert!(compile_pricing_variables(original.clone(), &[v.clone()]).is_err());
    v.integer = false;
    assert!(compile_pricing_variables(original.clone(), &[v.clone()]).is_ok());
    v.entries = vec![(0, 1.)];
    assert!(compile_pricing_variables(original.clone(), &[v.clone()]).is_err());
    v.entries.clear();
    v.cost = f64::MAX;
    v.arcs = vec![(0, 2.)];
    assert!(compile_pricing_variables(original.clone(), &[v]).is_err());
    assert_eq!(original.networks[0].graph.arcs[0].cost, 0.);
}

#[test]
fn formatted_registry_survives_resume_fixed_copies_and_tree_reporting() {
    let model = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_original_variables.json"
    )))
    .unwrap();
    let (problem, registry) = model.compile::<2>().unwrap();
    let root = solve(problem, options()).unwrap();
    let resumed = root.snapshot.solve(options()).unwrap();
    assert_eq!(
        resumed.master.lp, -1.,
        "resuming must not expand costs twice"
    );
    let selected = root.values.iter().position(|q| *q > 0.5).unwrap();
    let fixed = root
        .snapshot
        .fix_route(selected, 1.)
        .unwrap()
        .solve(options())
        .unwrap();
    let report = format::SolveResult::root(&fixed, 0.)
        .with_original_variables(&registry)
        .unwrap();
    assert_eq!(report.original_variables.len(), 1);
    assert_eq!(report.original_variables[0].lower, 1.);
    assert_eq!(report.original_variables[0].upper, 1.);
    let mut tree = ModelTreeOptions::default();
    tree.node = options();
    registry.configure_tree(&mut tree).unwrap();
    let tree = root.snapshot.solve_tree(tree).unwrap();
    let report = format::SolveResult::tree(&tree, 0.)
        .with_original_variables(&registry)
        .unwrap();
    assert_eq!(report.incumbent, Some(-1.));
    assert_eq!(report.original_variables[0].lower, 1.);
    assert_eq!(report.original_variables[0].upper, 1.);
}
