use super::*;

#[test]
fn routing_r201_starts_with_its_low_feasibility_penalty() {
    let mut instance = Instance::read(format!(
        "{}/instances/Solomon/R201.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    instance.truncate(100).unwrap();
    let graph = instance.graph(8, Rounding::Tenth).unwrap();
    let problem = model::Problem::set_partitioning(graph, 100.).unwrap();
    let mut options = parse_options("solve", vec![]).unwrap().unwrap().options;
    options.cg.max_iterations = 1;
    options.cg.time_limit = std::time::Duration::from_secs(5);
    let output = model::solve(problem, options).unwrap();
    assert_eq!(
        output.master.status,
        baldes::master::cg::Status::IterationLimit
    );
    assert!((output.master.lp - 276.).abs() < 1e-8);
    assert_eq!(output.master.bound, f64::NEG_INFINITY);
    // The routing adapter's policy must not replace the general model default.
    assert_eq!(model::Options::default().cg.artificial_cost, 1_200_000.);
}

#[test]
fn disabling_enumeration_disables_initial_gap_and_fixing_paths() {
    for strong in [false, true] {
        let mut args = vec!["--tree".to_owned(), "--no-enumeration".to_owned()];
        if strong {
            args.push("--strong-branching".to_owned());
        }
        let parsed = parse_options("solve", args).unwrap().unwrap();
        assert!(!parsed.enumerate);
        assert!(parsed.options.enumeration.is_none());
        assert!(!parsed.options.gap_enumeration);
        assert!(!parsed.options.reduced_cost_fixing);
        assert_eq!(parsed.strong_branching, strong);
    }
    let defaults = parse_options("solve", vec![]).unwrap().unwrap();
    assert!(defaults.options.enumeration.is_some());
    assert!(defaults.options.gap_enumeration);
    assert!(defaults.options.reduced_cost_fixing);
}

#[test]
fn enumerated_mip_cannot_silently_restore_disabled_enumeration() {
    for enable in ["--enumerated-mip", "--deluxing"] {
        let args = ["--tree", "--no-enumeration", enable]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let error = parse_options("solve", args).err().unwrap();
        assert!(error.contains("--no-enumeration"));
    }
}

#[test]
fn automatic_arc_instances_share_owner_metadata_within_each_network() {
    let problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    // The original full-model construction confirms one x owner per network,
    // independent of arc instance IDs, and distinct owners across networks.
    for mode in ["default", "arc", "resource", "cluster", "ryan-foster"] {
        let (variables, selection) = routing_branch_variables(&problem, mode);
        assert_eq!(selection.len(), variables.len());
        let mut owners = std::collections::BTreeMap::new();
        for variable in &variables {
            assert_eq!(variable.priority, 1.);
            let metadata = &selection[&variable.name];
            assert_eq!(metadata.rule, model::BranchSelectionRule::MostFractional);
            assert_eq!(metadata.variable_priority, 1.);
            assert_eq!(metadata.default_factor, 1.);
            let network = variable.networks[0].network;
            let owner = owners.entry(network).or_insert(&metadata.family);
            assert_eq!(*owner, &metadata.family);
        }
        assert_eq!(owners.len(), 2);
        assert_ne!(owners[&7], owners[&42]);
        for network in &problem.networks {
            assert!(
                variables
                    .iter()
                    .filter(|v| v.networks[0].network == network.id)
                    .count()
                    > 1
            );
        }
    }
    let single_network = model::Problem {
        rows: problem.rows.clone(),
        networks: vec![problem.networks[0].clone()],
    };
    let (_, selection) = routing_branch_variables(&single_network, "default");
    let owners: std::collections::BTreeSet<_> = selection.values().map(|s| &s.family).collect();
    assert_eq!(owners.len(), 1);
}

#[test]
fn edge_and_hybrid_producers_keep_their_existing_registration_boundary() {
    let problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let (edge, selection) = routing_branch_variables(&problem, "edge");
    assert!(!edge.is_empty());
    assert!(selection.is_empty());
    let (hybrid, selection) = routing_branch_variables(&problem, "hybrid");
    assert!(hybrid.is_empty());
    assert!(selection.is_empty());
}
