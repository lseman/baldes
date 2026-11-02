use super::*;

#[test]
fn multirow_knapsack_candidates_match_native_shared_tree() {
    for (case, line) in include_str!("multirow_native.txt").lines().enumerate() {
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let max_one_k = fields.next().unwrap().parse().unwrap();
        let max_cuts_per_round = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        let parameters = KnapsackParameters {
            max_cuts_per_round,
            use_rounding: true,
            max_one_k,
            violation_tolerance: 1e-6,
            ..KnapsackParameters::default()
        };
        let separator = KnapsackSeparator::with_parameters(parameters).unwrap();
        let rows: Vec<_> = (0..2)
            .map(|row| {
                let seed = if row == 0 || case % 4 == 0 {
                    case
                } else {
                    case + 7
                };
                let rhs = 5 + seed % 8;
                let totals = (0..=rhs)
                    .filter_map(|q| {
                        let n = (seed * 11 + q * 7 + seed * q * 3) % 9;
                        (n != 0)
                            .then_some((q as i32, n as f64 / if seed % 7 == 0 { 128. } else { 8. }))
                    })
                    .collect();
                (totals, rhs as i32)
            })
            .collect();
        let cuts = separator.separate_totals(&rows).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, (row, cut)) in cuts.iter().enumerate() {
            let native_row: usize = fields.next().unwrap().parse().unwrap();
            let rounding = fields.next().unwrap() == "1";
            let multiplier: u64 = fields.next().unwrap().parse().unwrap();
            let native_rhs: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let thresholds: Vec<i32> = (0..n)
                .map(|_| fields.next().unwrap().parse().unwrap())
                .collect();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(*row, native_row, "case {case}, candidate {index}");
            if rounding {
                assert!(matches!(cut.kind, KnapsackKind::Rounding));
                assert_eq!(
                    (cut.numerator as f64 / cut.denominator as f64).to_bits(),
                    multiplier,
                    "case {case}, candidate {index}"
                );
                assert!(thresholds.is_empty());
            } else {
                assert!(matches!(cut.kind, KnapsackKind::Steps));
                assert_eq!(thresholds.first(), Some(&0));
            }
            assert_eq!(cut.rhs, native_rhs, "case {case}, candidate {index}");
            assert_eq!(
                cut.thresholds,
                if rounding {
                    &thresholds[..]
                } else {
                    &thresholds[1..]
                },
                "case {case}, candidate {index}"
            );
            assert_eq!(
                cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn resource_knapsack_separation_matches_native_aggregated_columns() {
    let parameters = KnapsackParameters {
        max_cuts_per_round: 8,
        use_rounding: true,
        max_one_k: 8,
        violation_tolerance: 1e-6,
        ..KnapsackParameters::default()
    };
    let resources = vec![
        KnapsackResource {
            id: 0,
            variable_ids: vec![0],
        },
        KnapsackResource {
            id: 1,
            variable_ids: vec![1],
        },
    ];
    let graphs = vec![
        KnapsackGraph {
            id: 7,
            resources: resources.clone(),
        },
        KnapsackGraph { id: 8, resources },
    ];
    let mut separator = ResourceKnapsackSeparator {
        parameters,
        ..ResourceKnapsackSeparator::default()
    };
    for (case, line) in include_str!("resource_separate_native.txt")
        .lines()
        .enumerate()
    {
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let count: usize = fields.next().unwrap().parse().unwrap();
        let rows = vec![
            KnapsackPreparationRow {
                rhs: 6 + (case % 4) as i32,
                load_totals: BTreeMap::from([(2, 0.125000004), (4, 0.375000006)]),
                graph_resources: BTreeMap::from([(Some(0), (0, 1.25)), (Some(1), (1, 0.5))]),
            },
            KnapsackPreparationRow {
                rhs: 7 + (case % 5) as i32,
                load_totals: BTreeMap::from([(1, 0.250000004), (3, 0.500000006)]),
                graph_resources: BTreeMap::from([(Some(0), (1, 1.)), (Some(1), (0, 1.5))]),
            },
        ];
        separator.prepare(&graphs, &rows).unwrap();
        let columns: Vec<_> = (0..12)
            .map(|i| KnapsackColumn {
                graph_id: if i % 3 == 0 { 9 } else { 7 + (i % 2) as i32 },
                resources: vec![
                    ((case * 3 + i * 5) % 12) as f64 / 2. + if i % 4 == 0 { 0.0000008 } else { 0. },
                    ((case + i * 7) % 12) as f64 / 2. + if i % 4 == 1 { 0.0000012 } else { 0. },
                ],
                value: ((case * 7 + i * 3) % 9) as f64 / 8.
                    + if i % 2 == 0 { 0.000000004 } else { 0.000000006 },
            })
            .collect();
        let cuts = separator.separate(&columns).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, cut) in cuts.iter().enumerate() {
            let id: i32 = fields.next().unwrap().parse().unwrap();
            let row: usize = fields.next().unwrap().parse().unwrap();
            let rhs: i32 = fields.next().unwrap().parse().unwrap();
            let class: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                (cut.id, cut.row, cut.cut.rhs, cut.cut_class),
                (id, row, rhs, class),
                "case {case}, candidate {index}"
            );
            assert_eq!(
                cut.cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn resource_knapsack_coefficient_matches_native_scaled_route_resources() {
    let cut = ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: BTreeMap::from([(7, (0, 1.25)), (8, (1, 0.5))]),
        cut: RouteLoadKnapsackCut {
            kind: KnapsackKind::Steps,
            numerator: 1,
            denominator: 1,
            rhs: 9,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![],
            coefficients: BTreeMap::from([(0, 0), (1, 1), (3, 4), (6, 9)]),
        },
        dual: 0.,
    };
    for (case, line) in include_str!("coefficient_native.txt").lines().enumerate() {
        let mut fields = line.split_whitespace();
        let graph: i32 = fields.next().unwrap().parse().unwrap();
        let mode: usize = fields.next().unwrap().parse().unwrap();
        let first: f64 = fields.next().unwrap().parse().unwrap();
        let second: f64 = fields.next().unwrap().parse().unwrap();
        let expected: i32 = fields.next().unwrap().parse().unwrap();
        let resources = match mode {
            0 => vec![],
            1 => vec![first],
            _ => vec![first, second],
        };
        assert_eq!(
            cut.coefficient(graph, &resources),
            expected,
            "case {case}: {line}"
        );
        assert_eq!(
            cut.coefficient_mapped(graph, &resources, &BTreeMap::from([(0, 0), (1, 1)])),
            expected,
            "mapped case {case}: {line}"
        );
    }
}

#[test]
fn knapsack_preparation_matches_native_graph_mapping_and_failures() {
    let graphs = vec![
        KnapsackGraph {
            id: 7,
            resources: vec![
                KnapsackResource {
                    id: 0,
                    variable_ids: vec![10],
                },
                KnapsackResource {
                    id: 1,
                    variable_ids: vec![11],
                },
            ],
        },
        KnapsackGraph {
            id: 8,
            resources: vec![
                KnapsackResource {
                    id: 0,
                    variable_ids: vec![],
                },
                KnapsackResource {
                    id: 1,
                    variable_ids: vec![20],
                },
            ],
        },
        KnapsackGraph {
            id: 7,
            resources: vec![KnapsackResource {
                id: 0,
                variable_ids: vec![30],
            }],
        },
        KnapsackGraph {
            id: 9,
            resources: vec![KnapsackResource {
                id: 0,
                variable_ids: vec![-1],
            }],
        },
    ];
    let mut separator = ResourceKnapsackSeparator::default();
    for (case, line) in include_str!("prepare_native.txt").lines().enumerate() {
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), case);
        let ok = fields.next().unwrap() == "1";
        let count: usize = fields.next().unwrap().parse().unwrap();
        let mut rows = vec![
            KnapsackPreparationRow {
                rhs: 7,
                load_totals: BTreeMap::new(),
                graph_resources: BTreeMap::new()
            };
            if case == 1 {
                0
            } else if case == 0 || case == 7 {
                2
            } else {
                1
            }
        ];
        match case {
            0 => {
                rows[0]
                    .graph_resources
                    .extend([(Some(0), (0, 1.25)), (Some(1), (1, 0.5))]);
                rows[1].graph_resources.insert(Some(2), (0, 2.));
            }
            1 => (),
            2 => {
                rows[0].graph_resources.insert(None, (0, 1.));
            }
            3 => {
                rows[0].graph_resources.insert(Some(0), (2, 1.));
            }
            4 => {
                rows[0].graph_resources.insert(Some(1), (0, 1.));
            }
            5 => {
                rows[0].graph_resources.insert(Some(3), (0, 1.));
            }
            6 => {
                rows[0]
                    .graph_resources
                    .extend([(Some(0), (0, 1.25)), (Some(2), (0, 2.))]);
            }
            7 => {
                rows[0].graph_resources.insert(Some(0), (1, 3.));
                rows[1].graph_resources.insert(Some(1), (0, 1.));
            }
            _ => unreachable!(),
        }
        assert_eq!(separator.prepare(&graphs, &rows).is_ok(), ok, "case {case}");
        assert_eq!(separator.rows.len(), count, "case {case}");
        for (index, row) in separator.rows.iter().enumerate() {
            let n: usize = fields.next().unwrap().parse().unwrap();
            let expected: BTreeMap<i32, (usize, u64)> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        (
                            fields.next().unwrap().parse().unwrap(),
                            fields.next().unwrap().parse().unwrap(),
                        ),
                    )
                })
                .collect();
            let actual: BTreeMap<_, _> = row
                .graph_resources
                .iter()
                .map(|(&g, &(r, s))| (g, (r, s.to_bits())))
                .collect();
            assert_eq!(actual, expected, "case {case}, row {index}");
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn knapsack_preparation_matches_native_registered_graph_metadata() {
    let mut separator = ResourceKnapsackSeparator::default();
    for (case, line) in include_str!("prepare_registered_graph_native.txt")
        .lines()
        .enumerate()
    {
        let mut fields = line.split_whitespace();
        let graph_id: i32 = fields.next().unwrap().parse().unwrap();
        let associated: usize = fields.next().unwrap().parse().unwrap();
        let unassociated: usize = fields.next().unwrap().parse().unwrap();
        let variable: i32 = fields.next().unwrap().parse().unwrap();
        let ok = fields.next().unwrap() == "1";
        let row_count: usize = fields.next().unwrap().parse().unwrap();
        let graphs = vec![KnapsackGraph {
            id: graph_id,
            resources: vec![
                KnapsackResource {
                    id: unassociated,
                    variable_ids: vec![],
                },
                KnapsackResource {
                    id: associated,
                    variable_ids: vec![variable],
                },
            ],
        }];
        let input = [KnapsackPreparationRow {
            rhs: 0,
            load_totals: BTreeMap::new(),
            graph_resources: BTreeMap::from([(
                Some(0),
                (if case == 0 { associated } else { unassociated }, 1.25),
            )]),
        }];
        assert_eq!(
            separator.prepare(&graphs, &input).is_ok(),
            ok,
            "case {case}"
        );
        assert_eq!(separator.rows.len(), row_count);
        for row in &separator.rows {
            let count: usize = fields.next().unwrap().parse().unwrap();
            let expected: BTreeMap<i32, (usize, u64)> = (0..count)
                .map(|_| {
                    let id = fields.next().unwrap().parse().unwrap();
                    let resource = fields.next().unwrap().parse().unwrap();
                    let scale: f64 = fields.next().unwrap().parse().unwrap();
                    (id, (resource, scale.to_bits()))
                })
                .collect();
            let actual: BTreeMap<_, _> = row
                .graph_resources
                .iter()
                .map(|(&g, &(r, s))| (g, (r, s.to_bits())))
                .collect();
            assert_eq!(actual, expected, "case {case}");
        }
        assert!(fields.next().is_none());
    }
}

#[test]
fn knapsack_separation_matches_native_master_generated_columns() {
    let parameters = KnapsackParameters {
        max_cuts_per_round: 10,
        use_rounding: true,
        max_one_k: 1,
        ..KnapsackParameters::default()
    };
    let mut separator = ResourceKnapsackSeparator {
        parameters,
        ..ResourceKnapsackSeparator::default()
    };
    for (case, line) in include_str!("master_native.txt").lines().enumerate() {
        let mut fields = line.split_whitespace();
        let rhs: i32 = fields.next().unwrap().parse().unwrap();
        let count: usize = fields.next().unwrap().parse().unwrap();
        let columns: Vec<_> = (0..count)
            .map(|_| KnapsackColumn {
                graph_id: fields.next().unwrap().parse().unwrap(),
                value: fields.next().unwrap().parse().unwrap(),
                resources: vec![
                    fields.next().unwrap().parse().unwrap(),
                    fields.next().unwrap().parse().unwrap(),
                ],
            })
            .collect();
        separator.rows = vec![KnapsackRow::new(7, 1, 1., rhs)];
        let count: usize = fields.next().unwrap().parse().unwrap();
        let cuts = separator.separate(&columns).unwrap();
        assert_eq!(cuts.len(), count, "case {case}");
        for (index, cut) in cuts.iter().enumerate() {
            let id: i32 = fields.next().unwrap().parse().unwrap();
            let row: usize = fields.next().unwrap().parse().unwrap();
            let rhs: i32 = fields.next().unwrap().parse().unwrap();
            let class: i32 = fields.next().unwrap().parse().unwrap();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let coefficients: BTreeMap<i32, i32> = (0..n)
                .map(|_| {
                    (
                        fields.next().unwrap().parse().unwrap(),
                        fields.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                (cut.id, cut.row, cut.cut.rhs, cut.cut_class),
                (id, row, rhs, class),
                "case {case}, candidate {index}"
            );
            assert_eq!(cut.graph_resources, BTreeMap::from([(7, (1, 1.))]));
            assert_eq!(
                cut.cut.coefficients, coefficients,
                "case {case}, candidate {index}"
            );
        }
        assert!(fields.next().is_none());
    }
}
