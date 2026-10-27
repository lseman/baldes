use super::*;
use crate::graph::coupled::{
    PairedCostNormalization, PairedResourceArc, PairedResourceMode, PiecewiseLinearFunction,
};

fn prepared(case: usize, control_points: bool) -> Graph<2> {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut source = corpus["cases"][case]["model"].clone();
    source["networks"][0]["front_program_domain"] = false.into();
    source["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain".into();
    if control_points {
        fn knots(function: &serde_json::Value) -> serde_json::Value {
            let pairs = function["pairs"].as_array().unwrap();
            let (mut y, mut x) = (pairs[0][0].as_f64().unwrap(), pairs[0][1].as_f64().unwrap());
            let mut points = vec![serde_json::json!([x, y])];
            for pair in &pairs[1..] {
                let end = pair[1].as_f64().unwrap();
                y += pair[0].as_f64().unwrap() * (end - x);
                x = end;
                points.push(serde_json::json!([x, y]));
            }
            serde_json::json!({"points":points})
        }
        let resource = &mut source["networks"][0]["coupled_resources"][0];
        resource["initial"] = knots(&resource["initial"]);
        for arc in resource["arcs"].as_array_mut().unwrap() {
            for key in ["arrival", "increment", "lower", "upper"] {
                arc[key] = knots(&arc[key]);
            }
        }
        resource["kind"] = "paired_piecewise_linear".into();
        resource.as_object_mut().unwrap().remove("policy");
    }
    crate::master::model::format::Model::from_json(&source.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap()
        .networks
        .remove(0)
        .graph
}

#[test]
fn paired_control_optimizer_generates_minimum_maximum_and_the_reference_row_target() {
    for case in [0, 18] {
        for control_points in [false, true] {
            let graph = prepared(case, control_points);
            for (maximize, terminal, expected) in [
                (false, None, 3.),
                (true, None, 8.),
                (false, Some(Window::new(4., 4.).unwrap()), 4.),
                (true, Some(Window::new(4., 4.).unwrap()), 4.),
            ] {
                let result = optimize_paired_control(
                    &graph,
                    &[1, 2],
                    0,
                    maximize,
                    terminal,
                    ControlSearchLimits::default(),
                )
                .unwrap();
                assert!(result.complete,"case={case} points={control_points} max={maximize} terminal={terminal:?} {result:?}");
                assert_eq!(result.dependent, Some(expected));
                let bound = result.lower_bound.unwrap();
                let objective = if maximize { -expected } else { expected };
                assert!(bound <= objective && objective - bound < 1e-8, "{result:?}");
                let control = result.control.unwrap();
                assert!(graph
                    .coupled_kernel()
                    .unwrap()
                    .certify_paired_control(&[1, 2], &control, &mut 100_000, None)
                    .unwrap()
                    .unwrap()
                    .certifies_value(expected));
            }
        }
    }
}

fn nonlinear() -> Graph<2> {
    let mut graph = Graph::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let function = |points| PiecewiseLinearFunction::new(points).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            initial: function(vec![[0., 2.], [1., 2.]]),
            initial_base: Window::new(0., 0.).unwrap(),
            arcs: vec![PairedResourceArc {
                arrival: function(vec![[0., 1.], [2., 3.]]),
                increment: function(vec![[0., 2.], [2., 2.]]),
                lower: function(vec![[1., 0.], [2., 0.], [3., 3.]]),
                upper: function(vec![[1., 10.], [3., 10.]]),
                waiting: -1,
            }],
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
        }])
        .unwrap();
    graph
}

#[test]
fn paired_control_optimizer_covers_pwl_cells_signed_waiting_and_interior_clamp_crossings() {
    let graph = nonlinear();
    for (maximize, expected) in [(false, 2.25), (true, 4.)] {
        let result = optimize_paired_control(
            &graph,
            &[0],
            0,
            maximize,
            None,
            ControlSearchLimits::default(),
        )
        .unwrap();
        assert!(result.complete, "{result:?}");
        assert_eq!(result.dependent, Some(expected));
        let objective = if maximize { -expected } else { expected };
        assert!(
            result.lower_bound.unwrap() <= objective
                && objective - result.lower_bound.unwrap() < 1e-8
        );
        if !maximize {
            assert_eq!(result.control.unwrap().base, vec![0., 2.75]);
        }
    }
}

#[test]
fn paired_control_optimizer_proves_empty_regions_without_trusting_lp_status() {
    let graph = nonlinear();
    for terminal in [Window::new(5., 6.).unwrap(), Window::new(-2., -1.).unwrap()] {
        let result = optimize_paired_control(
            &graph,
            &[0],
            0,
            false,
            Some(terminal),
            ControlSearchLimits::default(),
        )
        .unwrap();
        assert!(result.complete, "{result:?}");
        assert!(result.control.is_none());
        assert!(result.dependent.is_none());
    }
}

#[test]
fn paired_control_optimizer_limits_preserve_unknown_and_partial_witnesses() {
    let graph = prepared(0, false);
    for limits in [
        ControlSearchLimits {
            max_cells: 0,
            ..Default::default()
        },
        ControlSearchLimits {
            max_work: 0,
            ..Default::default()
        },
        ControlSearchLimits {
            max_lp_iterations: 0,
            ..Default::default()
        },
        ControlSearchLimits {
            deadline: Some(Instant::now()),
            ..Default::default()
        },
    ] {
        let result = optimize_paired_control(&graph, &[1, 2], 0, true, None, limits).unwrap();
        assert!(!result.complete);
        assert!(result.lower_bound.is_none());
    }
    let result = optimize_paired_control(
        &graph,
        &[1, 2],
        0,
        true,
        None,
        ControlSearchLimits {
            max_cells: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!result.complete);
    assert!(result.lower_bound.is_none());
    assert_eq!(result.dependent, Some(8.));
}

#[test]
fn paired_control_optimizer_visits_more_than_four_thousand_source_cells() {
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    let mut arcs = Vec::new();
    for vertex in 0..4 {
        graph.add_arc(vertex, vertex + 1, 0.).unwrap();
        let split_constant = |value| {
            PiecewiseLinearFunction::new(vec![[0., value], [0.5, value], [1., value]]).unwrap()
        };
        arcs.push(PairedResourceArc {
            arrival: PiecewiseLinearFunction::new(vec![[0., 0.], [0.5, 0.5], [1., 1.]]).unwrap(),
            increment: split_constant(1.),
            lower: split_constant(0.),
            upper: split_constant(100.),
            waiting: 0,
        });
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            initial: PiecewiseLinearFunction::new(vec![[0., 0.], [1., 0.]]).unwrap(),
            initial_base: Window::new(0., 1.).unwrap(),
            arcs,
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
        }])
        .unwrap();

    let result = optimize_paired_control(
        &graph,
        &[0, 1, 2, 3],
        0,
        false,
        None,
        ControlSearchLimits {
            max_work: 10_000_000,
            max_lp_iterations: 100_000,
            ..Default::default()
        },
    )
    .unwrap();
    // The caller work budget, rather than a hidden 4,096-cell ceiling, stops
    // this larger Cartesian source partition. Its incumbent remains useful,
    // but no complete bound may be claimed after budget exhaustion.
    assert!(!result.complete, "{result:?}");
    assert!(result.cells > 4_096, "{result:?}");
    assert!(result.lower_bound.is_none());
    assert_eq!(result.dependent, Some(4.));
}

#[test]
fn paired_control_optimizer_supports_singleton_initial_domains_and_empty_paths() {
    let mut graph = prepared(0, false);
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    if let CoupledResource::SlopePaired {
        initial,
        initial_base,
        ..
    } = &mut definitions[0]
    {
        *initial =
            crate::graph::coupled::PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.)])
                .unwrap();
        *initial_base = Window::new(0., 0.).unwrap();
    }
    graph.set_coupled_resources(&definitions).unwrap();
    for (path, maximize, expected) in [
        (&[][..], false, 2.),
        (&[][..], true, 2.),
        (&[1, 2][..], false, 5.),
        (&[1, 2][..], true, 8.),
    ] {
        let result = optimize_paired_control(
            &graph,
            path,
            0,
            maximize,
            None,
            ControlSearchLimits::default(),
        )
        .unwrap();
        assert!(result.complete, "{result:?}");
        assert_eq!(result.dependent, Some(expected));
    }
}

#[test]
fn paired_control_optimizer_unsupported_arithmetic_and_input_errors_do_not_forge_emptiness() {
    let mut graph = nonlinear();
    assert!(optimize_paired_control(&graph, &[0], 99, false, None, Default::default()).is_err());
    assert!(optimize_paired_control(&graph, &[99], 0, false, None, Default::default()).is_err());
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    if let CoupledResource::PairedPiecewiseLinear { initial, .. } = &mut definitions[0] {
        *initial = PiecewiseLinearFunction::new(vec![[0., 1e308], [2., 1e308]]).unwrap();
    }
    graph.set_coupled_resources(&definitions).unwrap();
    let result = optimize_paired_control(&graph, &[0], 0, false, None, Default::default()).unwrap();
    assert!(!result.complete && result.control.is_none() && result.lower_bound.is_none());
    let mut graph = prepared(0, false);
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    if let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] {
        *policy = crate::graph::coupled::FrontOperationPolicy::FrontAlgebra;
    }
    graph.set_coupled_resources(&definitions).unwrap();
    graph.set_front_program_domain(true).unwrap();
    assert!(optimize_paired_control(&graph, &[1, 2], 0, true, None, Default::default()).is_err());
}

#[test]
fn paired_control_optimizer_bounds_match_independent_rational_one_arc_extrema() {
    use crate::master::numerics::ExactSum;
    use std::cmp::Ordering;
    // Fractions are kept as small integer numerator/positive denominator pairs.
    // The oracle enumerates endpoints and analytical growth/clamp crossings,
    // independently of the LP-cell compiler and its dual bounds.
    for waiting in [-3, -1, 0, 1, 3] {
        for levels in [[0, 0, 0], [0, 6, 0], [4, 0, 4], [0, 3, 1], [2, 1, 5]] {
            let mut graph = nonlinear();
            let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
            if let CoupledResource::PairedPiecewiseLinear { arcs, .. } = &mut definitions[0] {
                arcs[0].waiting = waiting;
                arcs[0].lower = PiecewiseLinearFunction::new(vec![
                    [1., levels[0] as f64],
                    [3., levels[1] as f64],
                    [5., levels[2] as f64],
                ])
                .unwrap();
                arcs[0].upper = PiecewiseLinearFunction::new(vec![[1., 20.], [5., 20.]]).unwrap();
            }
            graph.set_coupled_resources(&definitions).unwrap();
            let mut candidates: Vec<(i128, i128)> = vec![];
            for (i, x) in [1_i128, 3, 5].into_iter().enumerate() {
                candidates.push((
                    (4 + i128::from(waiting) * (x - 1)).max(levels[i] as i128),
                    1,
                ));
            }
            for (i, start) in [1_i128, 3].into_iter().enumerate() {
                let delta = (levels[i + 1] - levels[i]) as i128;
                let mut denominator = 2 * i128::from(waiting) - delta;
                if denominator == 0 {
                    continue;
                }
                let mut numerator =
                    2 * levels[i] as i128 - delta * start - 8 + 2 * i128::from(waiting);
                if denominator < 0 {
                    numerator = -numerator;
                    denominator = -denominator;
                }
                if start * denominator <= numerator && numerator <= (start + 2) * denominator {
                    candidates.push((
                        (4 - i128::from(waiting)) * denominator + i128::from(waiting) * numerator,
                        denominator,
                    ));
                }
            }
            candidates.sort_by(|a, b| (a.0 * b.1).cmp(&(b.0 * a.1)));
            for maximize in [false, true] {
                let (mut numerator, denominator) = if maximize {
                    *candidates.last().unwrap()
                } else {
                    candidates[0]
                };
                if maximize {
                    numerator = -numerator;
                }
                let result =
                    optimize_paired_control(&graph, &[0], 0, maximize, None, Default::default())
                        .unwrap();
                assert!(
                    result.complete,
                    "waiting={waiting} lower={levels:?} {result:?}"
                );
                let mut lower = ExactSum::default();
                lower
                    .product(result.lower_bound.unwrap(), denominator as f64)
                    .unwrap();
                assert_ne!(
                    lower.compare(numerator as f64).unwrap(),
                    Ordering::Greater,
                    "{result:?}"
                );
                let signed = if maximize {
                    -result.dependent.unwrap()
                } else {
                    result.dependent.unwrap()
                };
                let mut upper = ExactSum::default();
                upper.product(signed, denominator as f64).unwrap();
                assert_ne!(
                    upper.compare(numerator as f64).unwrap(),
                    Ordering::Less,
                    "{result:?}"
                );
                assert!(
                    (numerator as f64 / denominator as f64) - result.lower_bound.unwrap() < 1e-7
                );
            }
        }
    }
}

#[test]
fn signed_paired_objectives_require_selected_controls_in_certificates_and_master_pricing() {
    for points in [false, true] {
        let mut graph = prepared(0, points);
        graph
            .set_coupled_objective_coefficients(&[(0, -2.)])
            .unwrap();
        assert!(graph.set_front_program_domain(true).is_err());
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.has_negative_paired_objectives());
        assert!(!graph.front_program_domain());
        let root = kernel.root(true);
        assert!(kernel.dominance_penalty(&root, &root, true).is_none());
        let mut paths = crate::enumeration::PathStorage::new(1).unwrap();
        assert!(paths.inspect(&graph, f64::INFINITY, &[], &[]).is_err());
        assert!(kernel
            .certify_with_controls(&[1, 2], &[], &mut 1_000_000, None)
            .is_none());
        let optimum = optimize_paired_control(
            &graph,
            &[1, 2],
            0,
            true,
            None,
            ControlSearchLimits::default(),
        )
        .unwrap();
        assert!(optimum.complete && optimum.dependent == Some(8.));
        let controls = [optimum.control.unwrap()];
        let certificate = kernel
            .certify_with_controls(&[1, 2], &controls, &mut 1_000_000, None)
            .unwrap();
        assert_eq!(certificate, (true, [-16., -16.]));
        let numerical = kernel
            .replay_with_controls(&[1, 2], &controls)
            .unwrap()
            .unwrap();
        assert_eq!(numerical.cost, -16.);
        let mut solver =
            crate::pricing::Solver::new(graph.vertices.len(), graph.resources.len()).unwrap();
        solver.graph = graph;
        assert!(solver
            .price_phase(0, 10, false)
            .unwrap_err()
            .to_string()
            .contains("source-control master pricing"));
        solver
            .graph
            .set_coupled_objective_coefficients(&[(0, 0.)])
            .unwrap();
        assert!(!solver
            .graph
            .coupled_kernel()
            .unwrap()
            .has_negative_paired_objectives());
    }
}

#[test]
fn terminal_domain_shares_budgets_and_does_not_promote_rounded_sums_to_singletons() {
    for points in [false, true] {
        let mut graph = prepared(0, points);
        let mut limits = ControlSearchLimits::default();
        let variable = terminal_domain(&graph, &[1, 2], 0, &mut limits)
            .unwrap()
            .unwrap();
        assert!(variable.lower <= 3. && variable.upper >= 8. && variable.lower != variable.upper);
        assert!(
            limits.max_work < ControlSearchLimits::default().max_work
                && limits.max_lp_iterations < ControlSearchLimits::default().max_lp_iterations
        );
        let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
        match &mut definitions[0] {
            CoupledResource::SlopePaired { arcs, .. } => {
                for arc in arcs {
                    let (lo, hi) = arc.increment.domain();
                    arc.increment =
                        crate::graph::coupled::PreparedSlopeFunction::from_prepared_pairs(vec![
                            (0.5, lo),
                            (0., hi),
                        ])
                        .unwrap();
                }
            }
            CoupledResource::PairedPiecewiseLinear { arcs, .. } => {
                for arc in arcs {
                    let lo = arc.increment.points.first().unwrap()[0];
                    let hi = arc.increment.points.last().unwrap()[0];
                    arc.increment =
                        PiecewiseLinearFunction::new(vec![[lo, 0.5], [hi, 0.5]]).unwrap();
                }
            }
            _ => unreachable!(),
        }
        graph.set_coupled_resources(&definitions).unwrap();
        let fixed = terminal_domain(&graph, &[1, 2], 0, &mut ControlSearchLimits::default())
            .unwrap()
            .unwrap();
        assert_eq!((fixed.lower, fixed.upper), (1., 1.));
        assert!(terminal_domain(
            &graph,
            &[1, 2],
            0,
            &mut ControlSearchLimits {
                max_work: 0,
                ..Default::default()
            }
        )
        .is_none());
        assert!(terminal_domain(
            &graph,
            &[1, 2],
            0,
            &mut ControlSearchLimits {
                deadline: Some(Instant::now()),
                ..Default::default()
            }
        )
        .is_none());
        match &mut definitions[0] {
            CoupledResource::SlopePaired { arcs, .. } => {
                for (i, value) in [(1, 0.1), (2, 0.2)] {
                    let (lo, hi) = arcs[i].increment.domain();
                    arcs[i].increment =
                        crate::graph::coupled::PreparedSlopeFunction::from_prepared_pairs(vec![
                            (value, lo),
                            (0., hi),
                        ])
                        .unwrap();
                }
            }
            CoupledResource::PairedPiecewiseLinear { arcs, .. } => {
                for (i, value) in [(1, 0.1), (2, 0.2)] {
                    for p in &mut arcs[i].increment.points {
                        p[1] = value;
                    }
                }
            }
            _ => unreachable!(),
        }
        graph.set_coupled_resources(&definitions).unwrap();
        assert!(properties::invariant_value(&graph, 0, &[1, 2], &mut 1_000_000, None).is_none());
        if let Some(Some(domain)) =
            terminal_domain(&graph, &[1, 2], 0, &mut ControlSearchLimits::default())
        {
            assert!(domain.lower < domain.upper);
        }
    }
}

#[test]
fn exact_source_row_scaling_closes_attained_bounds_without_rounding_nonexact_rows() {
    use crate::master::{preprocessing::Bounds, RowSense};
    let run = |rows| {
        let cell = program::Cell {
            bounds: vec![
                Bounds {
                    lower: 0.,
                    upper: 8.
                };
                2
            ],
            rows,
            base_count: 1,
        };
        let mut budget = Budget {
            work: 1_000_000,
            iterations: 10_000,
            deadline: None,
        };
        match linear::solve_cell(&cell, false, &mut budget) {
            linear::CellResult::Bounded { bound, .. } => bound,
            _ => panic!("missing independently verified cell bound"),
        }
    };
    let bound = run(vec![
        program::row(vec![(0, -7.), (1, 7.)], RowSense::Equal, 0.),
        program::row(vec![(0, 1.)], RowSense::GreaterEqual, 3.),
    ]);
    assert_eq!(bound, 3.);
    // Both problems have exact minimum1/7. One has a nonrepresentable RHS
    // quotient; the other a nonrepresentable coefficient quotient. Neither
    // may become a stronger rounded constraint during row normalization.
    for rows in [
        vec![program::row(vec![(1, 7.)], RowSense::GreaterEqual, 1.)],
        vec![
            program::row(vec![(0, -1.), (1, 7.)], RowSense::GreaterEqual, 0.),
            program::row(vec![(0, 1.)], RowSense::GreaterEqual, 1.),
        ],
    ] {
        let bound = run(rows);
        assert!(bound > 0.);
        let product_upper = -crate::core::rounding::multiply_down(-bound, 7.).unwrap();
        assert!(product_upper <= 1., "bound {bound} exceeds exact1/7");
    }
}
