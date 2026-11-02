use super::*;

fn kernel_generator(priorities: &[f64]) -> ComponentClassGenerator {
    ComponentClassGenerator {
        name: "collection".into(),
        network: 7,
        priority: 1.,
        coordinates: priorities
            .iter()
            .enumerate()
            .map(|(id, &priority)| ComponentGeneratorCoordinate {
                coordinate: ComponentCoordinate {
                    id,
                    arc_values: vec![],
                },
                kind: ComponentVariableKind::Integer,
                priority,
                upper: f64::INFINITY,
                current_upper: f64::INFINITY,
            })
            .collect(),
        ordering_coordinates: vec![],
        priority_rule: ComponentPriorityRule::AddedDepth,
        minimum_fractionality: 0.02,
        integrality_tolerance: 1e-8,
    }
}

#[derive(serde::Deserialize)]
struct OrderCapture {
    kind: String,
    values: Vec<Vec<Option<i64>>>,
    priorities: Vec<f64>,
    native: serde_json::Value,
}

fn records(values: Vec<Vec<Option<i64>>>) -> Vec<integration::Record> {
    let n = values.len();
    values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let key = values
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| v.map(|v| (i, v)))
                .collect();
            (
                key,
                vec![n - index, index],
                Column {
                    quantity: 0.5,
                    remainder: 0.5,
                    fractional: true,
                    values,
                },
            )
        })
        .collect()
}

#[test]
fn recovered_collections_match_large_tolerant_chains_without_total_order_sorting() {
    let captures: Vec<OrderCapture> =
        serde_json::from_str(include_str!("component_order_capture.json")).unwrap();
    for capture in captures {
        let records = records(capture.values);
        let mut work = ordering::Work::new(None);
        if capture.kind == "stable" {
            let expected: Vec<usize> = serde_json::from_value(capture.native).unwrap();
            assert_eq!(
                integration::ordered_records(
                    &records,
                    (0..records.len()).collect(),
                    &[],
                    0,
                    &mut work
                )
                .unwrap()
                .unwrap(),
                expected
            );
        } else {
            let expected: Vec<(usize, Vec<usize>)> =
                serde_json::from_value(capture.native).unwrap();
            let columns: Vec<_> = records.iter().map(|r| &r.2).collect();
            let groups = kernel_generator(&capture.priorities)
                .groups(&columns, &vec![true; capture.priorities.len()], None)
                .unwrap();
            let actual: Vec<_> = groups
                .into_iter()
                .map(|(v, group)| {
                    let order = column_order(&group, v, &mut work).unwrap();
                    (
                        v,
                        order
                            .into_iter()
                            .map(|i| {
                                columns
                                    .iter()
                                    .position(|&c| std::ptr::eq(c, group[i]))
                                    .unwrap()
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }
}

#[derive(serde::Deserialize)]
struct TreeOrderCapture {
    values: Vec<Vec<Option<i64>>>,
    tree: Vec<Vec<(usize, String, f64)>>,
    native: Vec<usize>,
}

#[test]
fn inherited_class_partitions_precede_leaf_column_ordering() {
    let captures: Vec<TreeOrderCapture> =
        serde_json::from_str(include_str!("component_tree_order_capture.json")).unwrap();
    for capture in captures {
        let records = records(capture.values);
        let tree: Vec<_> = capture
            .tree
            .into_iter()
            .map(|bounds| ComponentSequence {
                bounds: bounds
                    .into_iter()
                    .map(|(coordinate, sense, threshold)| ComponentBound {
                        coordinate,
                        threshold,
                        sense: if sense == "G" {
                            ComponentSense::GreaterEqual
                        } else {
                            ComponentSense::LessEqual
                        },
                    })
                    .collect(),
            })
            .collect();
        assert_eq!(
            integration::ordered_records(
                &records,
                (0..records.len()).collect(),
                &tree.iter().collect::<Vec<_>>(),
                0,
                &mut ordering::Work::new(None)
            )
            .unwrap()
            .unwrap(),
            capture.native
        );
    }
}

#[derive(serde::Deserialize)]
struct PublicationInput {
    family: usize,
    outer: f64,
    rule: u8,
    priority: f64,
    level: i32,
}
#[derive(serde::Deserialize)]
struct PublicationCapture {
    quota: usize,
    inputs: Vec<PublicationInput>,
    native: Vec<usize>,
}

#[test]
fn shared_publication_preserves_family_level_coordinate_priority_quota_and_creation_order() {
    let captures: Vec<PublicationCapture> =
        serde_json::from_str(include_str!("component_publication_capture.json")).unwrap();
    assert_eq!(captures.len(), 120);
    for capture in captures {
        let keys: Vec<_> = capture
            .inputs
            .into_iter()
            .map(|i| ordering::PriorityKey {
                family: i.family,
                family_priority: i.outer,
                rule: i.rule,
                variable_priority: i.priority,
                added_level: i.level,
            })
            .collect();
        assert_eq!(
            ordering::publish(&keys, capture.quota, &mut ordering::Work::new(None)).unwrap(),
            capture.native
        );
    }
}

#[derive(serde::Deserialize)]
struct PostfilterCapture {
    priorities: Vec<f64>,
    rule: u8,
    published: Vec<usize>,
    selected: Vec<usize>,
}

#[test]
fn maximal_coordinate_priority_filter_retains_the_native_zero_baseline_and_band() {
    let captures: Vec<PostfilterCapture> =
        serde_json::from_str(include_str!("component_postfilter_capture.json")).unwrap();
    for capture in captures {
        let keys: Vec<_> = capture
            .published
            .iter()
            .map(|&i| ordering::PriorityKey {
                family: 0,
                family_priority: 1.,
                rule: capture.rule,
                variable_priority: capture.priorities[i],
                added_level: 1,
            })
            .collect();
        let selected: Vec<_> = ordering::publish(&keys, 3, &mut ordering::Work::new(None))
            .unwrap()
            .into_iter()
            .map(|i| capture.published[i])
            .collect();
        assert_eq!(selected, capture.selected);
    }
}

#[test]
fn interrupted_collection_ordering_returns_no_publication() {
    use std::cell::Cell;
    use std::time::Duration;
    let first = Cell::new(true);
    let mut indices: Vec<_> = (0..256).rev().collect();
    let mut work = ordering::Work::new(Some(Instant::now() + Duration::from_millis(1)));
    let completed = ordering::stable(
        &mut indices,
        &|a, b| {
            if first.replace(false) {
                std::thread::sleep(Duration::from_millis(2));
            }
            a < b
        },
        &mut work,
    );
    assert!(completed.is_none());
    let keys = [ordering::PriorityKey {
        family: 0,
        family_priority: 1.,
        rule: 0,
        variable_priority: 1.,
        added_level: 1,
    }];
    assert!(ordering::publish(&keys, 1, &mut work).is_none());
}

#[derive(serde::Deserialize)]
struct RoundingCapture {
    value_bits: u64,
    actual_bits: u64,
}

#[test]
fn recovered_fractional_rounding_preserves_adjacent_binary64_boundaries() {
    let captures: Vec<RoundingCapture> =
        serde_json::from_str(include_str!("component_rounding_capture.json")).unwrap();
    assert_eq!(captures.len(), 273);
    for capture in captures {
        // Preserve adjacent binary64 inputs independently of JSON's decimal parser.
        let value = f64::from_bits(capture.value_bits);
        assert_eq!(
            fraction(value),
            f64::from_bits(capture.actual_bits),
            "{value}"
        );
    }
}

#[test]
fn fractional_list_admission_flag_and_class_violation_have_distinct_thresholds() {
    let generator = ComponentClassGenerator {
        name: "thresholds".into(),
        network: 7,
        priority: 1.,
        coordinates: vec![],
        ordering_coordinates: vec![],
        priority_rule: ComponentPriorityRule::AddedDepth,
        minimum_fractionality: 0.02,
        integrality_tolerance: 1e-8,
    };
    assert!(generator.record(1. + 1e-6).unwrap().is_none());
    let (_, flagged) = generator.record(1. + 1.005e-6).unwrap().unwrap();
    assert!(!flagged);
    let (_, flagged) = generator.record(1. + 1.02e-6).unwrap().unwrap();
    assert!(flagged);
    assert!(generator.record(0.02).unwrap().unwrap().1);
    assert!(!generator.fractional(0.02));
    assert!(generator.fractional(0.020002));
}

#[derive(serde::Deserialize)]
struct Capture {
    name: String,
    rule: i32,
    kind: String,
    values: Vec<Vec<Option<i64>>>,
    quantities: Vec<f64>,
    prefix: Vec<ComponentBound>,
    #[serde(default)]
    tree: Vec<Vec<ComponentBound>>,
    candidates: Vec<Expected>,
}

#[derive(serde::Deserialize)]
struct Expected {
    level: i32,
    weight: f64,
    bounds: Vec<ComponentBound>,
}

#[test]
fn recovered_disaggregation_matches_recursive_sparse_and_inherited_classes() {
    let captures: Vec<Capture> =
        serde_json::from_str(include_str!("component_generator_capture.json")).unwrap();
    assert_eq!(captures.len(), 105);
    for capture in captures {
        let n = capture.values[0].len();
        let kind = if capture.kind == "B" {
            ComponentVariableKind::Binary
        } else {
            ComponentVariableKind::Integer
        };
        let generator = ComponentClassGenerator {
            name: capture.name.clone(),
            network: 7,
            priority: 1.,
            ordering_coordinates: vec![],
            coordinates: (0..n)
                .map(|id| ComponentGeneratorCoordinate {
                    coordinate: ComponentCoordinate {
                        id,
                        arc_values: vec![],
                    },
                    kind,
                    priority: 2. - id as f64,
                    current_upper: if kind == ComponentVariableKind::Binary {
                        1.
                    } else {
                        4.
                    },
                    upper: if kind == ComponentVariableKind::Binary {
                        1.
                    } else {
                        4.
                    },
                })
                .collect(),
            priority_rule: if capture.rule == 2 {
                ComponentPriorityRule::CoordinatePriority
            } else {
                ComponentPriorityRule::AddedDepth
            },
            minimum_fractionality: 0.02,
            integrality_tolerance: 1e-8,
        };
        let columns: Vec<_> = capture
            .values
            .into_iter()
            .zip(capture.quantities)
            .map(|(values, quantity)| Column {
                quantity,
                remainder: quantity,
                fractional: true,
                values,
            })
            .collect();
        let mut sequence = ComponentSequence {
            bounds: capture.prefix,
        };
        let filtered: Vec<_> = columns
            .iter()
            .filter(|c| {
                sequence
                    .bounds
                    .iter()
                    .all(|b| satisfies(b, c.values[b.coordinate].unwrap_or(0) as f64))
            })
            .collect();
        let mut actual = vec![];
        if capture.tree.is_empty() {
            generator
                .separate(
                    &filtered,
                    &mut vec![true; n],
                    &mut sequence,
                    0,
                    &mut actual,
                    None,
                )
                .unwrap();
        } else {
            let tree: Vec<_> = capture
                .tree
                .into_iter()
                .map(|bounds| ComponentSequence { bounds })
                .collect();
            generator
                .explore(
                    &filtered,
                    &mut vec![true; n],
                    &mut sequence,
                    &tree.iter().collect::<Vec<_>>(),
                    &mut actual,
                    None,
                )
                .unwrap();
        }
        assert_eq!(actual.len(), capture.candidates.len(), "{}", capture.name);
        for (actual, expected) in actual.iter().zip(capture.candidates) {
            assert_eq!(actual.level, expected.level, "{}", capture.name);
            assert_eq!(actual.sequence.bounds, expected.bounds, "{}", capture.name);
            assert_eq!(actual.class_weight, expected.weight, "{}", capture.name);
        }
    }
}

#[test]
fn disaggregation_keeps_explicit_zero_distinct_from_absent_coordinate() {
    let present = Column {
        quantity: 0.5,
        remainder: 0.5,
        fractional: true,
        values: vec![Some(0)],
    };
    let high = Column {
        values: vec![Some(3)],
        ..present
    };
    let absent = Column {
        quantity: 0.5,
        remainder: 0.5,
        fractional: true,
        values: vec![None],
    };
    assert_eq!(
        disaggregate(&[&high, &present, &present], 0, None)
            .unwrap()
            .unwrap(),
        [1.5, 0.]
    );
    assert_eq!(
        disaggregate(&[&high, &absent, &absent], 0, None)
            .unwrap()
            .unwrap(),
        [1.5]
    );
}

fn fractional_patterns() -> (Output<2>, Options, ComponentClassGenerator) {
    let mut graph = Graph::<2>::new(10, 0).unwrap();
    let mut entries = vec![];
    let mut coordinates = vec![vec![], vec![]];
    for i in 0..8 {
        graph
            .add_arc(0, i + 1, if i < 4 { 1. } else { 10. })
            .unwrap();
        graph.add_arc(i + 1, 9, 0.).unwrap();
        entries.push(vec![(i % 4, if i < 4 { 2. } else { 1. })]);
        entries.push(vec![]);
        for (v, values) in coordinates.iter_mut().enumerate() {
            values.push((i < 4 && i & (1 << v) != 0).then_some(1));
            values.push(None);
        }
    }
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 8.).unwrap(),
            arc_entries: entries,
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    let generator = ComponentClassGenerator {
        name: "patterns".into(),
        network: 7,
        priority: 1e9,
        coordinates: coordinates
            .into_iter()
            .enumerate()
            .map(|(i, arc_values)| ComponentGeneratorCoordinate {
                coordinate: ComponentCoordinate {
                    id: 11 * (i + 1),
                    arc_values,
                },
                kind: ComponentVariableKind::Binary,
                priority: 2. - i as f64,
                upper: 1.,
                current_upper: 1.,
            })
            .collect(),
        ordering_coordinates: vec![],
        priority_rule: ComponentPriorityRule::AddedDepth,
        minimum_fractionality: 0.02,
        integrality_tolerance: 1e-8,
    };
    (solve(problem, options.clone()).unwrap(), options, generator)
}

#[test]
fn automatic_generator_finds_a_conjunction_after_integer_single_coordinate_counts() {
    let (root, options, generator) = fractional_patterns();
    assert_eq!(root.master.lp, 2.);
    let proposals = generator.proposals(&root, 3, 1e-7, None).unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].value, 0.5);
    let model_strong::Restriction::GeneratedComponent { class, .. } = &proposals[0].restriction
    else {
        panic!("component class expected")
    };
    assert_eq!(
        class.sequence.bounds,
        [
            ComponentBound {
                coordinate: 11,
                threshold: 1.,
                sense: ComponentSense::GreaterEqual
            },
            ComponentBound {
                coordinate: 22,
                threshold: 1.,
                sense: ComponentSense::GreaterEqual
            },
        ]
    );
    let child = root
        .snapshot
        .with_component_bounds(
            class,
            crate::master::preprocessing::Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    assert_eq!(child.state.component_branches.len(), 1);
    assert_eq!(child.fork().unwrap().state.component_branches.len(), 1);
    assert!(root.snapshot.state.component_branches.is_empty());
    let solved = child.solve(options).unwrap();
    let mut reordered = generator.clone();
    reordered.coordinates.swap(0, 1);
    assert!(reordered.proposals(&solved, 3, 1e-7, None).is_ok());
    let mut changed = generator.clone();
    changed.coordinates[0].coordinate.arc_values[0] = Some(1);
    assert!(changed.proposals(&solved, 3, 1e-7, None).is_err());
}

#[test]
fn automatic_component_tree_preserves_the_integer_optimum() {
    let (root, options, generator) = fractional_patterns();
    let mut tree = ModelTreeOptions::default();
    tree.node = options;
    tree.branching_registry.branching = NetworkBranchingPolicy::Phased;
    tree.branching_registry.component_generators = vec![generator];
    let solved = root.snapshot.solve_tree(tree).unwrap();
    assert_eq!(solved.incumbent, 40.);
    assert_eq!(solved.status, crate::master::tree::TreeStatus::Optimal);
}

#[test]
fn imported_expansion_preserves_routing_class_rows_and_generator_history() {
    use crate::master::preprocessing::Bounds;
    let (root, options, generator) = fractional_patterns();
    let oracle = &root.snapshot.state.oracles[0];
    let problem = Problem {
        rows: root.snapshot.state.shared_rows.clone(),
        networks: vec![Network {
            id: 7,
            graph: oracle.network.graph.as_ref().clone(),
            multiplicity: oracle.total_multiplicity,
            arc_entries: oracle.network.arc_entries.as_ref().clone(),
            route_entries: oracle.network.route_entries.as_ref().clone(),
            activity_entries: oracle.activity_rows.clone(),
        }],
    };
    let column = |id| imported::Column {
        id,
        cost: 0.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: true,
        entries: vec![],
    };
    let imported = imported::solve(
        problem,
        vec![],
        vec![imported::Subproblem {
            id: 42,
            variables: vec![],
            multiplicity: Bounds {
                lower: 0.,
                upper: 1.,
            },
            columns: vec![column(0)],
        }],
        options.clone(),
    )
    .unwrap();
    let proposals = generator
        .proposals(&imported.output, 3, 1e-7, None)
        .unwrap();
    let model_strong::Restriction::GeneratedComponent { class, .. } = &proposals[0].restriction
    else {
        panic!("component class expected")
    };
    let node = imported
        .snapshot()
        .unwrap()
        .with_component_bounds(
            class,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    let before = node.solve(options.clone()).unwrap();
    let after = node
        .solve_with_columns(42, vec![column(1)], options)
        .unwrap();
    assert_eq!(before.output.master.lp, after.output.master.lp);
    assert_eq!(after.output.master.lp, 11.5);
    assert_eq!(after.output.snapshot.state.component_branches.len(), 1);
    assert_eq!(
        after.output.snapshot.state.component_branches[0].sequence,
        class.sequence
    );
    assert!(root.snapshot.state.component_branches.is_empty());
}
