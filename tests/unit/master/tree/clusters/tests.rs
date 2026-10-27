use super::*;
#[test]
fn mst_partition_and_aggregate_coefficients_match_physical_crossings() {
    let mut graph = crate::Graph::<2>::new(6, 0).unwrap();
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
    let p = RoutingContext::set_partitioning(graph, 4.).unwrap();
    let groups = partition(&p, &Default::default()).unwrap();
    assert_eq!(groups, vec![vec![1, 2], vec![3, 4]]);
    let nodes = vec![0, 1, 3, 2, 4, 5];
    let arcs = nodes
        .windows(2)
        .map(|pair| {
            p.model.networks[0]
                .graph
                .arcs
                .iter()
                .find(|a| a.tail == pair[0] && a.head == pair[1])
                .unwrap()
                .id
        })
        .collect();
    let route = Route {
        coupled_controls: Vec::new(),
        coupled_values: Vec::new(),
        nodes: nodes.iter().map(|&v| v as i32).collect(),
        arcs,
        cost: 0.,
        original_cost: 0.,
        resources: vec![],
        resource_profile: vec![],
    };
    let boundary = BranchConstraint::ClusterBoundary {
        vertices: vec![1, 2],
        sense: RowSense::LessEqual,
        rhs: 1.,
    };
    let pair = BranchConstraint::ClusterPair {
        first: groups[0].clone(),
        second: groups[1].clone(),
        sense: RowSense::LessEqual,
        rhs: 1.,
    };
    assert_eq!(boundary.coefficient(&p, &route), 2.);
    assert_eq!(pair.coefficient(&p, &route), 3.);
    let depot = BranchConstraint::ClusterPair {
        first: vec![0, 5],
        second: vec![1, 2],
        sense: RowSense::LessEqual,
        rhs: 1.,
    };
    assert_eq!(depot.coefficient(&p, &route), 1.);
    for groups in [
        vec![vec![1, 2], vec![2, 3, 4]],
        vec![vec![1, 2], vec![3]],
        vec![vec![0, 1, 2, 3, 4]],
        vec![vec![], vec![1, 2, 3, 4]],
    ] {
        assert!(partition(
            &p,
            &ClusterBranchingOptions {
                partition: Some(groups),
                ..Default::default()
            }
        )
        .is_err());
    }
    assert!(partition(
        &p,
        &ClusterBranchingOptions {
            theta: f64::NAN,
            ..Default::default()
        }
    )
    .is_err());
}
