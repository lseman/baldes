use super::*;

fn root(networks: usize) -> Output<2> {
    let mut problem = crate::master::model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let template = problem.networks[0].clone();
    for index in problem.networks.len()..networks {
        let mut network = template.clone();
        network.id = index as i32 + 100;
        network.multiplicity.upper = 0.;
        problem.networks.push(network);
    }
    let mut options = crate::master::model::Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    crate::master::model::solve(problem, options).unwrap()
}

fn proposal(network: i32, path: bool, priority: f64, value: f64) -> Proposal {
    let name = format!("{}:{network}", if path { "paths" } else { "multiplicity" });
    Proposal {
        history_key: name.clone(),
        name,
        priority,
        value,
        restriction: if path {
            Restriction::PathCount { network }
        } else {
            Restriction::Multiplicity { network }
        },
        history_eligible: path,
        fresh: true,
        origin: None,
        component_admission: None,
    }
}

#[test]
fn count_groups_merge_signed_zero_but_keep_distinct_nearby_levels() {
    let output = root(2);
    let mut options = ModelTreeOptions::default();
    options.branching_registry.paths_per_network = vec![
        PathsPerNetworkBranching {
            network: 7,
            priority: 0.,
        },
        PathsPerNetworkBranching {
            network: 42,
            priority: 0.,
        },
    ];
    let mut candidates = vec![
        proposal(42, false, -0., 1.5),
        proposal(42, true, 0., 1.4),
        proposal(7, false, 0., 1.5),
        proposal(7, true, -0., 1.5),
    ];
    let mut order = vec![0, 1, 2, 3];
    rank_count_slots(&candidates, &options, &output, &mut order);
    assert_eq!(order, [2, 3, 0, 1]);
    candidates[0].priority = 1e-8;
    candidates[1].priority = 1e-8;
    rank_count_slots(&candidates, &options, &output, &mut order);
    assert_eq!(order, [0, 1, 2, 3]);
}

#[test]
fn large_count_registry_preserves_network_registration_and_other_family_slots() {
    let output = root(129);
    let ids: Vec<_> = output
        .snapshot
        .state
        .oracles
        .iter()
        .map(|o| o.network.id)
        .collect();
    assert_eq!(ids.len(), 129);
    let mut options = ModelTreeOptions::default();
    options.branching_registry.paths_per_network = ids
        .iter()
        .rev()
        .map(|&network| PathsPerNetworkBranching {
            network,
            priority: -5.,
        })
        .collect();
    let mut candidates = vec![];
    for &network in ids.iter().rev() {
        candidates.push(proposal(network, false, -5., 1.5));
        candidates.push(proposal(network, true, -5., 1.5));
    }
    let mut unrelated = proposal(7, false, -5., 1.5);
    unrelated.name = "paths:custom-original-variable".into();
    unrelated.restriction = Restriction::Mapped(MappedBranchVariable {
        name: unrelated.name.clone(),
        priority: -5.,
        networks: vec![],
        masters: vec![],
    });
    candidates.insert(5, unrelated);
    let mut order: Vec<_> = (0..candidates.len()).collect();
    rank_count_slots(&candidates, &options, &output, &mut order);
    assert_eq!(order[5], 5);
    let inferred: Vec<_> = order
        .iter()
        .filter_map(|&index| {
            let candidate = &candidates[index];
            match candidate.restriction {
                Restriction::Multiplicity { network }
                    if candidate.name.starts_with("multiplicity:") =>
                {
                    Some(network)
                }
                _ => None,
            }
        })
        .collect();
    assert_eq!(inferred, ids);
    let paths: Vec<_> = order
        .iter()
        .filter_map(|&index| {
            let candidate = &candidates[index];
            match candidate.restriction {
                Restriction::PathCount { network } => Some(network),
                _ => None,
            }
        })
        .collect();
    assert_eq!(paths, ids.into_iter().rev().collect::<Vec<_>>());
}
