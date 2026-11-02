use super::*;

fn triangles() -> (Output<2>, Options, ComponentClassBranching) {
    triangles_with_integer_column(false)
}

fn triangles_with_integer_column(
    integer_column: bool,
) -> (Output<2>, Options, ComponentClassBranching) {
    triangles_with_resource_cost(integer_column, 0.)
}

fn triangles_with_resource_cost(
    integer_column: bool,
    resource_cost: f64,
) -> (Output<2>, Options, ComponentClassBranching) {
    let mut covers = vec![
        vec![0, 1],
        vec![0, 2],
        vec![1, 2],
        vec![3, 4],
        vec![3, 5],
        vec![4, 5],
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
    ];
    if integer_column {
        covers.extend([vec![6], vec![6]]);
    }
    let terminal = covers.len() + 1;
    let mut graph = Graph::<2>::new(terminal + 1, 0).unwrap();
    let mut arc_entries = vec![];
    for (route, rows) in covers.iter().enumerate() {
        graph
            .add_arc(
                0,
                route + 1,
                (if route < 6 || route == 12 { 1. } else { 2. })
                    + resource_cost * (route + 1) as f64,
            )
            .unwrap();
        graph.add_arc(route + 1, terminal, 0.).unwrap();
        arc_entries.push(rows.iter().map(|&row| (row, 1.)).collect());
        arc_entries.push(vec![]);
    }
    let problem = Problem {
        rows: (0..if integer_column { 7 } else { 6 })
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![Network {
            id: 42,
            graph,
            multiplicity: Window::new(0., if integer_column { 7. } else { 6. }).unwrap(),
            arc_entries,
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    let mut values = vec![None; 2 * covers.len()];
    values[0] = Some(1);
    let class = ComponentClassBranching {
        name: "selected pair".into(),
        priority: 1.,
        minimum_fractionality: 0.02,
        network: 42,
        coordinates: vec![ComponentCoordinate {
            id: 0,
            arc_values: values,
        }],
        sequence: ComponentSequence {
            bounds: vec![ComponentBound {
                coordinate: 0,
                threshold: 1.,
                sense: ComponentSense::GreaterEqual,
            }],
        },
    };
    (solve(problem, options.clone()).unwrap(), options, class)
}

#[test]
fn nonempty_component_emits_the_three_observed_native_rows() {
    let (root, _, class) = triangles();
    assert_eq!(root.master.lp, 3.);
    let value = class.snapshot_activity(&root.snapshot).unwrap();
    assert_eq!(value, 0.5);
    let children = class.branching_children(&root.snapshot, value).unwrap();
    assert_eq!(children.len(), 3);
    let rows: Vec<_> = children
        .iter()
        .map(|child| &child.state.component_local_rows[0])
        .collect();
    assert!(rows[0].lower);
    assert_eq!(rows[0].rhs, 1.);
    assert_eq!(rows[0].sequence, class.sequence);
    assert!(rows[1].lower);
    assert_eq!(rows[1].rhs, 3.);
    assert_eq!(rows[1].sequence, class.sequence.sibling().unwrap());
    assert!(!rows[2].lower);
    assert_eq!(rows[2].rhs, 2.);
    assert!(rows[2].sequence.bounds.is_empty());
    assert!(root.snapshot.state.component_local_rows.is_empty());
    assert!(root.snapshot.state.component_branches.is_empty());
    // Exhaustively cover the fixture's complete integer route universe.
    let covers = [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]];
    let mut feasible = 0;
    let mut optimum = usize::MAX;
    for mask in 0usize..1 << 12 {
        let mut coverage = [0; 6];
        let mut count = 0;
        for route in 0..12 {
            if mask & (1 << route) == 0 {
                continue;
            }
            count += 1;
            if route < 6 {
                for &row in &covers[route] {
                    coverage[row] += 1;
                }
            } else {
                coverage[route - 6] += 1;
            }
        }
        if coverage != [1; 6] || count > 6 {
            continue;
        }
        feasible += 1;
        let cost = (0..12)
            .filter(|&route| mask & (1 << route) != 0)
            .map(|route| if route < 6 { 1 } else { 2 })
            .sum::<usize>();
        optimum = optimum.min(cost);
        let selected = usize::from(mask & 1 != 0);
        assert!(selected >= 1 || count - selected >= 3 || count <= 2);
    }
    assert!(feasible > 0);
    assert_eq!(optimum, 6);
}

#[test]
fn fixed_cardinality_omits_total_child_and_deeper_sequence_adds_complement() {
    let (root, options, mut class) = triangles();
    let fixed = root
        .snapshot
        .with_multiplicity(
            42,
            Bounds {
                lower: 3.,
                upper: 3.,
            },
        )
        .unwrap()
        .solve(options)
        .unwrap();
    let value = class.snapshot_activity(&fixed.snapshot).unwrap();
    let children = class.branching_children(&fixed.snapshot, value).unwrap();
    assert_eq!(children.len(), 2);
    assert!(children
        .iter()
        .all(|child| child.state.component_local_rows[0].lower));
    let mut coordinate = class.coordinates[0].clone();
    coordinate.id = 1;
    class.coordinates.push(coordinate);
    class.sequence.bounds.push(ComponentBound {
        coordinate: 1,
        threshold: 1.,
        sense: ComponentSense::GreaterEqual,
    });
    let children = class.branching_children(&root.snapshot, 0.5).unwrap();
    assert_eq!(children.len(), 4);
    assert_eq!(
        children
            .iter()
            .map(|child| child.state.component_local_rows[0].sequence.bounds.len())
            .collect::<Vec<_>>(),
        [2, 2, 1, 0]
    );
}

#[test]
fn committed_siblings_and_parent_rows_survive_forks_without_becoming_local_rows() {
    let (root, _, class) = triangles();
    let mut children = class.branching_children(&root.snapshot, 0.5).unwrap();
    inherit_branch_rows(&root.snapshot, &mut children);
    for child in &mut children {
        child.state.node_depth = 1;
    }
    for child in &children {
        assert_eq!(child.state.component_branch_exclusions.len(), 2);
        assert!(!child
            .state
            .component_branch_exclusions
            .iter()
            .any(|r| r.equivalent(&child.state.component_local_rows[0])));
    }
    let parent = &children[0];
    let mut grandchildren = vec![parent.fork().unwrap(), parent.fork().unwrap()];
    inherit_branch_rows(parent, &mut grandchildren);
    for child in grandchildren {
        assert!(child.state.component_local_rows.is_empty());
        assert_eq!(child.state.component_branch_exclusions.len(), 3);
    }
    assert_eq!(parent.state.component_branch_exclusions.len(), 2);
}

#[test]
fn sibling_exclusions_skip_the_equivalent_row_but_preserve_other_owners() {
    let (mut root, _, class) = triangles();
    let original = class.branching_children(&root.snapshot, 0.5).unwrap();
    let row = original[0].state.component_local_rows[0].clone();
    root.snapshot.state.component_branch_exclusions = Arc::new(vec![row.clone()]);
    let children = class.branching_children(&root.snapshot, 0.5).unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(
        children[0].state.component_local_rows[0].sequence,
        class.sequence.sibling().unwrap()
    );
    let mut different = row;
    different.network = 7;
    root.snapshot.state.component_branch_exclusions = Arc::new(vec![different]);
    assert_eq!(
        class.branching_children(&root.snapshot, 0.5).unwrap().len(),
        3
    );
}

#[test]
fn reused_coordinate_id_with_a_different_mapping_does_not_exclude_a_branch() {
    let (mut root, _, class) = triangles();
    let original = class.branching_children(&root.snapshot, 0.5).unwrap();
    let mut other = class.clone();
    other.coordinates[0].arc_values[0] = None;
    other.coordinates[0].arc_values[2] = Some(1);
    root.snapshot.state.component_branch_exclusions =
        Arc::new(vec![original[0].state.component_local_rows[0].clone()]);
    let children = other.branching_children(&root.snapshot, 0.5).unwrap();
    assert_eq!(children.len(), 3);
    assert!(!children[0].state.component_local_rows[0]
        .equivalent(&original[0].state.component_local_rows[0]));
}

#[test]
fn duplicate_covered_groups_can_leave_one_or_zero_children() {
    let (mut root, _, class) = triangles();
    let original = class.branching_children(&root.snapshot, 0.5).unwrap();
    let rows: Vec<_> = original
        .iter()
        .map(|child| child.state.component_local_rows[0].clone())
        .collect();
    root.snapshot.state.component_branch_exclusions = Arc::new(rows[..2].to_vec());
    let children = class.branching_children(&root.snapshot, 0.5).unwrap();
    assert_eq!(children.len(), 1);
    assert!(children[0].state.component_local_rows[0]
        .sequence
        .bounds
        .is_empty());
    root.snapshot.state.component_branch_exclusions = Arc::new(rows);
    assert!(class
        .branching_children(&root.snapshot, 0.5)
        .unwrap()
        .is_empty());
}

#[test]
fn multiway_component_tree_proves_the_enumerated_two_triangle_optimum() {
    use crate::master::tree::{ModelTreeOptions, NetworkBranchingPolicy, StrongBranchingOptions};
    let (root, node, mut class) = triangles();
    class.priority = 1000.;
    for strong in [false, true] {
        let options = ModelTreeOptions {
            node: node.clone(),
            node_limit: 200,
            time_limit: std::time::Duration::from_secs(10),

            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.branching = NetworkBranchingPolicy::Phased;
                __branch_registry.component_classes = vec![class.clone()];
                __branch_registry.strong_branching =
                    strong.then(StrongBranchingOptions::default);
                __branch_registry
            },
            ..Default::default()
        };
        let result = root.snapshot.solve_tree(options).unwrap();
        assert!(
            (result.incumbent - 6.).abs() < 1e-7,
            "strong={strong}: {result:?}"
        );
        assert!(
            (result.bound - 6.).abs() < 1e-7,
            "strong={strong}: {result:?}"
        );
    }
}

#[test]
fn fractional_predicates_keep_the_valid_binary_count_split() {
    let (root, _, mut class) = triangles();
    class.sequence.bounds[0].threshold = 0.5;
    let children = class.branching_children(&root.snapshot, 0.5).unwrap();
    assert_eq!(children.len(), 2);
    assert!(!children[0].state.component_local_rows[0].lower);
    assert_eq!(children[0].state.component_local_rows[0].rhs, 0.);
    assert!(children[1].state.component_local_rows[0].lower);
    assert_eq!(children[1].state.component_local_rows[0].rhs, 1.);
}

#[test]
fn root_local_rows_are_not_visited_by_the_ancestor_child_scan() {
    let (mut root, _, class) = triangles();
    let original = class.branching_children(&root.snapshot, 0.5).unwrap();
    root.snapshot.state.component_local_rows = original[0].state.component_local_rows.clone();
    let mut child = vec![root.snapshot.fork().unwrap()];
    inherit_branch_rows(&root.snapshot, &mut child);
    assert!(child[0].state.component_branch_exclusions.is_empty());
    assert!(child[0].state.component_local_rows.is_empty());
}

#[test]
fn integer_lp_columns_contribute_to_the_observed_native_partition_weights() {
    use crate::master::tree::{ModelTreeOptions, NetworkBranchingPolicy, StrongBranchingOptions};
    let (root, node, mut class) = triangles_with_integer_column(true);
    assert_eq!(root.master.lp, 4.);
    let value = class.snapshot_activity(&root.snapshot).unwrap();
    assert_eq!(value, 0.5);
    let children = class.branching_children(&root.snapshot, value).unwrap();
    let descriptors: Vec<_> = children
        .iter()
        .map(|child| {
            let row = &child.state.component_local_rows[0];
            (row.lower, row.rhs)
        })
        .collect();
    assert_eq!(descriptors, [(true, 1.), (true, 4.), (false, 3.)]);
    class.priority = 1000.;
    let result = root
        .snapshot
        .solve_tree(ModelTreeOptions {
            node,
            time_limit: std::time::Duration::from_secs(10),
            node_limit: 200,

            branching_registry: {
                let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                __branch_registry.branching = NetworkBranchingPolicy::Phased;
                __branch_registry.component_classes = vec![class];
                __branch_registry.strong_branching = Some(StrongBranchingOptions::default());
                __branch_registry
            },
            ..Default::default()
        })
        .unwrap();
    assert!((result.incumbent - 7.).abs() < 1e-7, "{result:?}");
    assert!((result.bound - 7.).abs() < 1e-7, "{result:?}");
}

#[test]
fn resource_activity_classes_match_original_multiway_rows_and_integer_oracle() {
    use crate::master::tree::{ModelTreeOptions, NetworkBranchingPolicy, StrongBranchingOptions};
    #[derive(serde::Deserialize)]
    struct Fixture {
        threshold: f64,
        activity: f64,
        rows: Vec<(bool, f64)>,
    }
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("component_resource_partition_fixtures.json")).unwrap();
    let (root, node, mut class) = triangles_with_resource_cost(true, 0.01);
    assert!((root.master.lp - 4.235).abs() < 1e-8);
    class.coordinates[0].arc_values = (0..28)
        .map(|arc| (arc % 2 == 0).then_some((arc / 2 + 1) as i64))
        .collect();
    for fixture in fixtures {
        class.sequence.bounds[0].threshold = fixture.threshold;
        let activity = class.snapshot_activity(&root.snapshot).unwrap();
        assert!((activity - fixture.activity).abs() < 1e-8);
        let children = class.branching_children(&root.snapshot, activity).unwrap();
        let rows: Vec<_> = children
            .iter()
            .map(|child| {
                let row = &child.state.component_local_rows[0];
                (row.lower, row.rhs)
            })
            .collect();
        assert_eq!(rows, fixture.rows);
    }
    class.sequence.bounds[0].threshold = 6.;
    class.priority = 1000.;
    for strong in [false, true] {
        let result = root
            .snapshot
            .solve_tree(ModelTreeOptions {
                node: node.clone(),
                time_limit: std::time::Duration::from_secs(10),
                node_limit: 200,

                branching_registry: {
                    let mut __branch_registry = crate::master::model::BranchingRegistry::default();
                    __branch_registry.branching = NetworkBranchingPolicy::Phased;
                    __branch_registry.component_classes = vec![class.clone()];
                    __branch_registry.strong_branching =
                        strong.then(StrongBranchingOptions::default);
                    __branch_registry
                },
                ..Default::default()
            })
            .unwrap();
        assert!(
            (result.incumbent - 7.39).abs() < 1e-7,
            "strong={strong}: {result:?}"
        );
        assert!(
            (result.bound - 7.39).abs() < 1e-7,
            "strong={strong}: {result:?}"
        );
    }
}
