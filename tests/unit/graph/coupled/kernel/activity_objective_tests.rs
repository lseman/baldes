use super::*;

fn graph() -> Graph<2> {
    let mut graph = Graph::new(4, 1).unwrap();
    graph.elementary = true;
    graph.vertices[0].elements.clear();
    graph.vertices[3].elements.clear();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, -20.), (1, 3, 0.), (0, 2, -20.), (2, 3, 0.)] {
        let id = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![1.; 4],
            weights: vec![3., 4., 0., 1.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    graph
}

#[test]
fn absolute_objective_and_pricing_coefficients_keep_intrinsic_activities() {
    for coefficient in [-2., -0.5, 0., 0.5, 1., 2.] {
        for adjustment in [0., -1.5] {
            let mut graph = graph();
            graph
                .set_coupled_objective_coefficients(&[(0, coefficient)])
                .unwrap();
            graph
                .set_coupled_pricing_adjustments(&[(0, adjustment)])
                .unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            for (path, activity) in [([0, 1], 11.), ([2, 3], 2.)] {
                assert_eq!(kernel.accumulated_activity(0, &path).unwrap(), activity);
                assert_eq!(kernel.replay(&path).unwrap(), Some(coefficient * activity));
                assert_eq!(
                    kernel.replay_pricing(&path).unwrap(),
                    Some((coefficient + adjustment) * activity)
                );
                for split in 0..=2 {
                    let (f, _) = kernel.replay_state(&path[..split]).unwrap().unwrap();
                    let mut b = kernel.root(false);
                    for &arc in path[split..].iter().rev() {
                        b = kernel
                            .extend_internal(
                                &b,
                                arc,
                                false,
                                false,
                                false,
                                &mut CoupledWorkspace::default(),
                            )
                            .unwrap()
                            .unwrap()
                            .0;
                    }
                    assert_eq!(
                        kernel.join_objective_costs(&f, &b).unwrap(),
                        Some((
                            (coefficient + adjustment) * activity,
                            coefficient * activity
                        ))
                    );
                }
            }
            for direction in [0., 1., 2.] {
                let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
                solver.graph = graph.clone();
                solver
                    .set_option("BALDESuseBidirectionalSearch", direction)
                    .unwrap();
                let output = solver.price_phase(0, 100, false).unwrap();
                assert!(output.exact);
                let values = [
                    -20. + (coefficient + adjustment) * 11.,
                    -20. + (coefficient + adjustment) * 2.,
                ];
                assert_eq!(output.minimum_reduced_cost, values[0].min(values[1]));
                for route in output.columns {
                    let activity = if route.arcs == [0, 1] { 11. } else { 2. };
                    assert_eq!(route.original_cost, -20. + coefficient * activity);
                }
            }
        }
    }
}

#[test]
fn a_tiny_nonzero_absolute_coefficient_is_not_rounded_away_as_q_minus_one() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 1e308,
            increments: vec![0.],
            weights: vec![1.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    graph
        .set_coupled_objective_coefficients(&[(0, 1e-308)])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(1e-308 - 1., -1.);
    assert_eq!(kernel.accumulated_activity(0, &[0]).unwrap(), 1e308);
    let cost = kernel.replay(&[0]).unwrap().unwrap();
    assert!(cost > 0.9 && cost < 1.1);
    assert_eq!(kernel.objective_coefficient(0), Some(1e-308));
    #[cfg(feature = "clp")]
    {
        let (valid, [lower, upper]) = kernel.certify(&[0]).unwrap();
        assert!(valid && lower <= cost && cost <= upper);
    }
}

#[cfg(feature = "clp")]
#[test]
fn zero_objective_coefficients_and_phase_one_do_not_suppress_row_activities() {
    let mut graph = graph();
    graph
        .set_coupled_objective_coefficients(&[(0, 0.)])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(0.));
    assert_eq!(
        kernel.certify_accumulated_activity(0, &[0, 1], &mut 100_000, None),
        Some((true, [11., 11.]))
    );
    graph.suppress_coupled_costs();
    graph.set_coupled_pricing_adjustments(&[(0, -1.)]).unwrap();
    assert_eq!(graph.coupled_path_cost(&[0, 1]).unwrap(), Some(0.));
    assert_eq!(
        graph.coupled_pricing_path_cost(&[0, 1]).unwrap(),
        Some(-11.)
    );
}

#[cfg(feature = "clp")]
#[test]
fn paired_associated_coefficients_keep_unit_raw_pricing_and_weighted_original_costs() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    for case_index in [0, 18] {
        for coefficient in [-2., -1., 0., 0.5, 1., 2., 1e-308] {
            let mut problem = crate::master::model::format::Model::from_json(
                &corpus["cases"][case_index]["model"].to_string(),
            )
            .unwrap()
            .problem::<2>()
            .unwrap();
            let graph = &mut problem.networks[0].graph;
            graph
                .set_coupled_objective_coefficients(&[(0, coefficient)])
                .unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let expected = coefficient * 3.;
            assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(expected));
            assert_eq!(
                kernel.selected_paired_values(&[1, 2]).unwrap(),
                Some(vec![(0, 3.)])
            );
            assert_eq!(kernel.objective_coefficient(0), Some(coefficient));
            assert_eq!(kernel.replay_pricing(&[1, 2]).unwrap(), Some(3.));
            assert_eq!(
                kernel.certify_pricing_with_budget(&[1, 2], &mut 100_000, None),
                Some((true, [3., 3.]))
            );
            let (feasible, interval) = kernel.certify(&[1, 2]).unwrap();
            assert!(feasible && interval[0] <= expected && expected <= interval[1]);
            for split in 0..=2 {
                let path = [1, 2];
                let mut f = kernel.root(true);
                for &arc in &path[..split] {
                    f = kernel.extend(&f, arc, true).unwrap().unwrap().0;
                }
                let mut b = kernel.root(false);
                for &arc in path[split..].iter().rev() {
                    b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
                }
                assert_eq!(
                    kernel.join_objective_costs(&f, &b).unwrap(),
                    Some((3., expected))
                );
            }
            let mut suppressed = kernel.clone();
            suppressed.suppress_costs();
            assert_eq!(suppressed.replay(&[1, 2]).unwrap(), Some(0.));
            assert_eq!(
                suppressed.selected_paired_values(&[1, 2]).unwrap(),
                Some(vec![(0, 3.)])
            );
        }
    }
}
