use super::*;
use crate::inputs::{Instance, Rounding};

fn two_cluster_graph() -> Graph<2> {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    for tail in 0..5 {
        for head in 1..6 {
            if tail == head {
                continue;
            }
            let cost = if tail > 0 && head < 5 && (tail <= 2) != (head <= 2) {
                10.
            } else {
                1.
            };
            graph
                .add_arc_with_id(100 + 7 * graph.arcs.len() as i32, tail, head, cost)
                .unwrap();
        }
    }
    graph
}

fn prepared_arcs(clusters: &NetworkClusterBranching, name: &str) -> Vec<i32> {
    clusters
        .prepared
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, arcs)| arcs.iter().map(|(id, _)| *id).collect())
        .unwrap_or_default()
}

#[test]
fn mst_partition_and_coefficients_count_physical_crossings() {
    let graph = two_cluster_graph();
    let mut clusters = NetworkClusterBranching::new(3);
    assert_eq!(
        clusters.groups(&graph).unwrap(),
        vec![vec![1, 2], vec![3, 4]]
    );
    clusters.prepare(std::iter::once((3, &graph))).unwrap();
    // Groups: 0 = depot {0, 5}, 1 = {1, 2}, 2 = {3, 4}.
    let nodes = [0usize, 1, 3, 2, 4, 5];
    let route: Vec<i32> = nodes
        .windows(2)
        .map(|pair| {
            graph
                .arcs
                .iter()
                .find(|a| a.tail == pair[0] && a.head == pair[1])
                .unwrap()
                .id
        })
        .collect();
    let count = |name: &str| {
        let arcs = prepared_arcs(&clusters, name);
        route.iter().filter(|id| arcs.contains(id)).count()
    };
    assert_eq!(count("cluster:3:1"), 2);
    assert_eq!(count("cluster:3:1:2"), 3);
    assert_eq!(count("cluster:3:0:1"), 1);
    // The depot boundary is the number of nonempty routes.
    assert_eq!(count("cluster:3:0"), 1);
    assert!(clusters.prepare(std::iter::once((4, &graph))).is_err());
}

#[test]
fn invalid_partitions_and_parameters_are_rejected() {
    let graph = two_cluster_graph();
    for groups in [
        vec![vec![1, 2], vec![2, 3, 4]],
        vec![vec![1, 2], vec![3]],
        vec![vec![0, 1, 2, 3, 4]],
        vec![vec![], vec![1, 2, 3, 4]],
    ] {
        let mut clusters = NetworkClusterBranching::new(0);
        clusters.partition = Some(groups);
        assert!(clusters.groups(&graph).is_err());
    }
    let mut clusters = NetworkClusterBranching::new(0);
    clusters.partition = Some(vec![vec![4, 3], vec![2, 1]]);
    assert_eq!(
        clusters.groups(&graph).unwrap(),
        vec![vec![1, 2], vec![3, 4]]
    );
    clusters.partition = None;
    clusters.theta = -1.;
    assert!(clusters.groups(&graph).is_err());
    clusters.theta = 1.;
    clusters.priority = f64::NAN;
    assert!(clusters.prepare(std::iter::once((0, &graph))).is_err());
}

#[test]
fn cluster_branching_tree_reaches_the_default_branching_optimum() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(25).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    let problem = Problem::set_partitioning(graph, 25.).unwrap();
    let solve = |clusters: Vec<NetworkClusterBranching>| {
        let mut options = ModelTreeOptions::default();
        options.branching_registry.mapped_variables = problem.arc_flow_variables(1.);
        options.branching_registry.clusters = clusters;
        options.node_limit = 2000;
        crate::master::tree::solve(problem.clone(), options).unwrap()
    };
    let default = solve(vec![]);
    let clustered = solve(vec![NetworkClusterBranching::new(0)]);
    // The root is fractional, so both trees branch.
    assert!(clustered.processed_nodes > 1);
    assert_eq!(default.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(clustered.status, crate::master::tree::TreeStatus::Optimal);
    assert!(
        (default.incumbent - clustered.incumbent).abs() < 1e-6,
        "{} {}",
        default.incumbent,
        clustered.incumbent
    );
}
