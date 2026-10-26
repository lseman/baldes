use super::*;
use baldes::cuts::{CutClass, CutRegistry, MemoryKind, RankOneCut};
use baldes::master::preprocessing::Bounds;
use model::{EnumerationOptions, MasterVariable, RegisteredCutCoefficients};

fn cut(vertices: usize, support: &[usize]) -> RankOneCut {
    let mut registry = CutRegistry::default();
    registry
        .add_strong_k_path_cut(vertices, support, 1, 0.)
        .unwrap();
    registry.cuts.remove(0)
}
fn gap_network(id: i32, first: usize, parallel: bool) -> Network<2> {
    let mut graph = Graph::<2>::new(5, 1).unwrap();
    for (a, b, c) in [
        (0, 1, 1.),
        (1, 3, 0.),
        (3, 4, 1.),
        (1, 2, 1.),
        (2, 3, 0.),
        (1, 4, 1.),
        (0, 3, 1.),
    ] {
        graph.add_arc(a, b, c).unwrap();
    }
    if parallel {
        graph.add_arc(0, 1, 1.5).unwrap();
    }
    let arc_entries = graph
        .arcs
        .iter()
        .map(|a| match a.head {
            1 => vec![(first, 1.)],
            3 => vec![(first + 1, 1.)],
            _ => vec![],
        })
        .collect();
    Network {
        id,
        graph,
        multiplicity: Window::new(0., 3.).unwrap(),
        arc_entries,
        route_entries: vec![],
    }
}
fn gap_problem(parallel: bool) -> Problem<2> {
    Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![gap_network(7, 0, parallel), gap_network(42, 2, parallel)],
    }
}
fn verify_descendants(problem: Problem<2>, descriptor: RankOneCut, row: Row, expected: f64) {
    let root = model::solve(problem, options()).unwrap();
    optimal(&root, 4.);
    let coefficients = [7, 42].map(|network| RegisteredCutCoefficients {
        network,
        cut: descriptor.clone(),
    });
    let child = root
        .snapshot
        .with_registered_cut_row(row.clone(), &coefficients)
        .unwrap();
    let solved = child.solve(options()).unwrap();
    optimal(&solved, expected);
    let pair = root
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1, 2])
        .unwrap();
    let fixed = child.fix_route(pair, 1.).unwrap();
    optimal(&fixed.solve(options()).unwrap(), expected);
    let reordered = root
        .snapshot
        .fix_route(pair, 1.)
        .unwrap()
        .with_registered_cut_row(row, &coefficients)
        .unwrap();
    optimal(&reordered.solve(options()).unwrap(), expected);
    let complete = fixed
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), expected);
    #[cfg(feature = "highs")]
    {
        let mip = complete
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(mip.mip.solution.unwrap().objective, expected);
    }
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
}

#[test]
fn registered_covering_ceilings_count_singletons_and_shift_fixed_rows() {
    let mut descriptor = cut(4, &[1, 2]);
    descriptor.visit_once = false;
    descriptor.class = CutClass::Covering;
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    verify_descendants(
        problem,
        descriptor,
        Row {
            sense: RowSense::GreaterEqual,
            rhs: 3.,
        },
        6.,
    );
}

#[test]
fn registered_limited_memory_resets_are_priced_and_replayed_consistently() {
    let mut descriptor = cut(5, &[1, 3]);
    descriptor.visit_once = false;
    descriptor.class = CutClass::Packing;
    descriptor.memory[2] = false;
    verify_descendants(
        gap_problem(false),
        descriptor,
        Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        },
        5.,
    );
}

#[test]
fn registered_arc_weights_preserve_parallel_arc_identity() {
    let mut descriptor = cut(5, &[1]);
    descriptor.visit_once = false;
    descriptor.class = CutClass::Packing;
    descriptor.weights.fill(0);
    descriptor.memory_kind = MemoryKind::Arc;
    descriptor.arc_memory = (0..8).collect();
    descriptor.arc_weights = [(0, 1), (2, 1)].into_iter().collect();
    verify_descendants(
        gap_problem(true),
        descriptor,
        Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        },
        4.5,
    );
}

#[test]
fn registered_strong_kpath_uses_its_limited_memory_entry_coefficient() {
    let mut descriptor = cut(5, &[1, 3]);
    descriptor.memory[2] = false;
    verify_descendants(
        gap_problem(false),
        descriptor,
        Row {
            sense: RowSense::GreaterEqual,
            rhs: 3.,
        },
        5.,
    );
}

#[test]
fn registered_graph_mapping_and_unbounded_master_coupling_survive_forks() {
    let mut descriptor = cut(4, &[1, 2]);
    descriptor.visit_once = false;
    descriptor.class = CutClass::Packing;
    descriptor.graph_support = [(7, vec![(1, 1), (2, 1)]), (42, vec![(1, 1)])]
        .into_iter()
        .collect();
    descriptor.graph_dependent_memory = true;
    descriptor.graph_memories = [(7, vec![true; 4]), (42, vec![true; 4])]
        .into_iter()
        .collect();
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let master = MasterVariable {
        id: 100,
        cost: 0.25,
        bounds: Bounds {
            lower: 0.,
            upper: f64::INFINITY,
        },
        integer: true,
        entries: vec![],
    };
    let root = model::solve_with_master_variables(problem, vec![master], options()).unwrap();
    let coefficients = [7, 42].map(|network| RegisteredCutCoefficients {
        network,
        cut: descriptor.clone(),
    });
    let linked = root
        .snapshot
        .with_registered_cut_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coefficients,
            &[(100, -1.)],
        )
        .unwrap();
    let solved = linked.solve(options()).unwrap();
    optimal(&solved, 4.25);
    assert_eq!(solved.master_values, vec![(100, 1.)]);
    let zero = linked
        .with_master_variable_bounds(
            100,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    optimal(&zero.solve(options()).unwrap(), 6.);
    let complete = zero.enumerate(&EnumerationOptions::default()).unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 6.);
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
}

#[test]
fn unsupported_registered_contracts_fail_before_kernel_preparation() {
    let root = model::solve(gap_problem(false), options()).unwrap();
    let base = cut(5, &[1, 3]);
    let mut invalid = base.clone();
    invalid.denominator = 8;
    assert!(root
        .snapshot
        .with_registered_cut_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &[RegisteredCutCoefficients {
                network: 7,
                cut: invalid
            }]
        )
        .is_err());
    let mut invalid = base.clone();
    invalid.visit_once = false;
    invalid.graph_pricing_support = [(7, vec![(1, 1)])].into_iter().collect();
    assert!(root
        .snapshot
        .with_registered_cut_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &[RegisteredCutCoefficients {
                network: 7,
                cut: invalid
            }]
        )
        .is_err());
    assert!(root
        .snapshot
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &[model::RankOneCoefficients {
                network: 7,
                weights: vec![0, 1, 0, 1, 0],
                denominator: 8
            }]
        )
        .is_err());
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
}

#[test]
fn registered_vertex_and_arc_carries_share_a_consistent_pricing_coefficient() {
    let mut descriptor = cut(5, &[1, 3]);
    descriptor.visit_once = false;
    descriptor.class = CutClass::Packing;
    descriptor.arc_weights = [(0, 1), (2, 1)].into_iter().collect();
    let master = MasterVariable {
        id: 100,
        cost: 1.,
        bounds: Bounds {
            lower: 0.,
            upper: f64::INFINITY,
        },
        integer: false,
        entries: vec![],
    };
    let root =
        model::solve_with_master_variables(gap_problem(false), vec![master], options()).unwrap();
    let coefficients = [7, 42].map(|network| RegisteredCutCoefficients {
        network,
        cut: descriptor.clone(),
    });
    let child = root
        .snapshot
        .with_registered_cut_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &coefficients,
            &[(100, -1.)],
        )
        .unwrap();
    let solved = child.solve(options()).unwrap();
    // Each covered customer contributes two increments, one vertex and one
    // arc. Dividing by two gives four units across both networks, so y=4.
    optimal(&solved, 8.);
    assert_eq!(solved.master_values, vec![(100, 4.)]);
    let complete = child.enumerate(&EnumerationOptions::default()).unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 8.);
}

#[test]
fn registered_type_two_uses_packing_semantics_in_master_and_pricing() {
    let mut descriptor = cut(5, &[1, 3]);
    descriptor.visit_once = false;
    descriptor.type2 = true;
    descriptor.class = CutClass::Covering;
    verify_descendants(
        gap_problem(false),
        descriptor,
        Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        },
        6.,
    );
}

#[test]
fn composite_shared_row_sums_same_network_terms_through_fixing_and_enumeration() {
    let root = model::solve(gap_problem(false), options()).unwrap();
    optimal(&root, 4.);
    let terms: Vec<_> = [7, 42]
        .into_iter()
        .flat_map(|network| {
            (0..2).map(move |_| RegisteredCutCoefficients {
                network,
                cut: {
                    let mut descriptor = cut(5, &[1, 3]);
                    descriptor.visit_once = false;
                    descriptor.class = CutClass::Packing;
                    descriptor
                },
            })
        })
        .collect();
    let row = Row {
        sense: RowSense::LessEqual,
        rhs: 2.,
    };
    // The original single-descriptor API still rejects accidental duplicates.
    assert!(root
        .snapshot
        .with_registered_cut_row(row.clone(), &terms)
        .is_err());
    let child = root
        .snapshot
        .with_composite_cut_row(row.clone(), &terms)
        .unwrap();
    optimal(&child.solve(options()).unwrap(), 6.);
    let pair = root
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs == [0, 1, 2])
        .unwrap();
    let fixed = child.fix_route(pair, 1.).unwrap();
    optimal(&fixed.solve(options()).unwrap(), 6.);
    let reordered = root
        .snapshot
        .fix_route(pair, 1.)
        .unwrap()
        .with_composite_cut_row(row, &terms)
        .unwrap();
    optimal(&reordered.solve(options()).unwrap(), 6.);
    let complete = fixed.enumerate(&EnumerationOptions::default()).unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 6.);
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
}

#[test]
fn composite_terms_share_one_master_variable_coefficient() {
    let master = MasterVariable {
        id: 100,
        cost: 0.25,
        bounds: Bounds {
            lower: 0.,
            upper: 4.,
        },
        integer: false,
        entries: vec![],
    };
    let root =
        model::solve_with_master_variables(gap_problem(false), vec![master], options()).unwrap();
    let terms: Vec<_> = [7, 42]
        .into_iter()
        .flat_map(|network| {
            (0..2).map(move |_| RegisteredCutCoefficients {
                network,
                cut: {
                    let mut descriptor = cut(5, &[1, 3]);
                    descriptor.visit_once = false;
                    descriptor.class = CutClass::Packing;
                    descriptor
                },
            })
        })
        .collect();
    let child = root
        .snapshot
        .with_composite_cut_master_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &terms,
            &[(100, -1.)],
        )
        .unwrap();
    let solved = child.solve(options()).unwrap();
    optimal(&solved, 5.);
    assert_eq!(solved.master_values, vec![(100, 4.)]);
    let complete = child.enumerate(&EnumerationOptions::default()).unwrap();
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    optimal(&complete.solve(certified).unwrap(), 5.);
}
