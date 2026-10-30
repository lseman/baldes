use super::*;

#[test]
fn rank_one_order_sorts_unrounded_values_largest_first() {
    let values = [0.25, 0.75, 0.5];
    let mut order = vec![0, 1, 2];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![1, 2, 0]);
}

#[test]
fn rank_one_order_keeps_cut_ids_within_tolerance() {
    // Distinct after 1e-8 rounding, but within the native 1e-6 tie window.
    let values = [0.5, 0.5000004, 0.4999997];
    let mut order = vec![2, 1, 0];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![0, 1, 2]);
}

#[test]
fn rank_one_order_is_signed_not_absolute() {
    let values = [-0.75, 0.5];
    let mut order = vec![0, 1];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![1, 0]);
}

fn cut_lifecycle_graph(fixture: &serde_json::Value, paired: bool) -> Graph<2> {
    use crate::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use crate::graph::coupled::{CoupledResource, MachineArcFunctionInput, PairedResourceMode};
    let network = &fixture["model"]["networks"][0];
    let mut graph = Graph::<2>::new(5, if paired { 0 } else { 1 }).unwrap();
    graph.graph_id = 7;
    if !paired {
        for vertex in 0..5 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
    }
    let mut paired_arcs = vec![];
    let function = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.5)]).unwrap()
    };
    for arc in network["arcs"].as_array().unwrap() {
        let id = graph
            .add_arc(
                arc["tail"].as_u64().unwrap() as usize,
                arc["head"].as_u64().unwrap() as usize,
                arc["cost"].as_f64().unwrap(),
            )
            .unwrap();
        let consumption = arc["consumption"][0].as_f64().unwrap();
        if paired {
            paired_arcs.push(function(consumption, 0.));
        } else {
            graph.arc_mut(id).unwrap().consumption[0] = consumption;
        }
    }
    if paired {
        // Match the successful paired public control. Integer unit demands
        // make base capacity 2.5 admit exactly singleton and pair routes.
        graph
            .set_coupled_resources(&[CoupledResource::PairedFront {
                mode: PairedResourceMode::CostUnbounded,
            }])
            .unwrap();
        let bounds = crate::graph::Window::new(0., 2.5).unwrap();
        let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
        let inputs: Vec<_> = paired_arcs
            .iter()
            .enumerate()
            .map(|(physical_arc, base)| MachineArcFunctionInput {
                physical_arc,
                tail_bounds: bounds,
                waiting_bounds: bounds,
                waiting_rate: 0,
                base,
                dependent: affine(0.),
                dependent_lower: affine(0.),
                dependent_upper: affine(8.),
            })
            .collect();
        let result = graph.prepare_paired_front_program(
            0,
            &inputs,
            affine(0.),
            bounds,
            bounds,
            FrontierBudget::default(),
        );
        assert!(
            matches!(result, FrontierOutcome::Complete(())),
            "{result:?}"
        );
    }
    graph
}

#[test]
fn generated_rank_three_cut_reaches_master_duals_and_modern_pricing_cache() {
    for paired in [false, true] {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-generated-rank-three.json"
        ))
        .unwrap();
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-generated-rank-three-reference.json"
        ))
        .unwrap();
        let paired_reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-paired-cut-lifecycle-reference.json"
        ))
        .unwrap();
        let reference = if paired {
            &paired_reference["rank_one"]
        } else {
            &reference
        };
        let graph = cut_lifecycle_graph(&fixture, paired);
        let mut options = RootOptions::default().with_rank_one_cuts();
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        options.cg.artificial_cost = 1000.;
        let output = super::super::solve(
            RoutingContext::set_partitioning(graph, 3.).unwrap(),
            options,
        )
        .unwrap();
        // Independent oracle: sum of the three unit cover rows requires 3/2
        // capacity-two routes. Each pair coefficient is floor(2/2)=1, and the
        // rank-three packing row allows at most one such route. A pair plus a
        // singleton attains the resulting lower bound of two vehicles.
        assert_eq!(output.lp_before_cuts, 1.5);
        assert_eq!(output.master.lp, 2.);
        let snapshot = output.snapshot.as_ref().unwrap();
        assert_eq!(snapshot.active_cuts(), 1);
        let cut = &snapshot.pricing.cuts.cuts[0];
        assert_eq!(cut.denominator, 2);
        assert_eq!(cut.rhs, 1);
        assert_eq!(cut.weights, vec![0, 1, 1, 1, 0]);
        assert_eq!(cut.class, crate::cuts::CutClass::Packing);
        let row = snapshot.cut.rows[0].unwrap();
        assert_eq!(snapshot.master.rhs[row], 1.);
        assert_eq!(snapshot.master.senses[row], RowSense::LessEqual);
        assert_eq!(cut.dual, -1.);
        assert_eq!(output.duals[row], -1.);
        for (column, stored) in snapshot.warm.columns.iter().enumerate() {
            let Some(stored) = stored else {
                continue;
            };
            let route = &snapshot.routes[stored.tag as usize];
            let visits = route.nodes.len() - 2;
            let expected = (visits / 2) as f64;
            let actual = snapshot
                .master
                .column_entries(column)
                .unwrap()
                .iter()
                .find(|(r, _)| *r == row)
                .map_or(0., |(_, value)| *value);
            assert_eq!(actual, expected);
        }
        assert_eq!(reference["support"], serde_json::json!([0, 1, 2]));
        assert_eq!(reference["numerators"], serde_json::json!([1, 1, 1]));
        assert_eq!(reference["pricing_value"], 1.);
        assert_eq!(reference["denominator"], cut.denominator);
        let mut pricing = snapshot.pricing.fork_for_node();
        pricing.price_phase(0, 20, false).unwrap();
        let inputs = &pricing.engine.as_ref().unwrap().cuts.inputs;
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].denominator, 2);
        assert_eq!(inputs[0].vertex_coefficient, vec![0, 1, 1, 1, 0]);
        assert_eq!(
            inputs[0].dual.to_bits(),
            reference["pricing_value"].as_f64().unwrap().to_bits()
        );
    }
}

#[test]
fn generated_strong_kpath_cut_reaches_master_duals_and_modern_pricing_cache() {
    for paired in [false, true] {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-generated-rank-three.json"
        ))
        .unwrap();
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-generated-kpath-reference.json"
        ))
        .unwrap();
        let paired_reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-paired-cut-lifecycle-reference.json"
        ))
        .unwrap();
        let reference = if paired {
            &paired_reference["strong_k_path"]
        } else {
            &reference
        };
        let graph = cut_lifecycle_graph(&fixture, paired);
        let mut options =
            RootOptions::default().with_kpath_cuts(KPathCutConfig::new(vec![0., 1., 1., 1.], 2.));
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        options.cg.artificial_cost = 1000.;
        let output = super::super::solve(
            RoutingContext::set_partitioning(graph, 3.).unwrap(),
            options,
        )
        .unwrap();
        // At most two unit-demand customers fit on a route. The generated set's
        // entry-count inequality therefore needs two routes; a pair and a singleton
        // attain the bound. Every available nonempty route enters this set once.
        assert_eq!(output.lp_before_cuts, 1.5);
        assert_eq!(output.master.lp, 2.);
        let snapshot = output.snapshot.as_ref().unwrap();
        assert_eq!(snapshot.active_cuts(), 1);
        let index = snapshot.cut.kpath_indices[0];
        let cut = &snapshot.pricing.cuts.cuts[index];
        assert!(cut.visit_once);
        assert_eq!(cut.rhs as f64, reference["rhs"].as_f64().unwrap());
        assert_eq!(cut.weights, vec![0, 1, 1, 1, 0]);
        let row = snapshot.cut.rows[index].unwrap();
        assert_eq!(snapshot.master.rhs[row], 2.);
        assert_eq!(snapshot.master.senses[row], RowSense::GreaterEqual);
        assert_eq!(cut.dual, 1.);
        assert_eq!(output.duals[row], 1.);
        for (column, stored) in snapshot.warm.columns.iter().enumerate() {
            let Some(_) = stored else {
                continue;
            };
            let actual = snapshot
                .master
                .column_entries(column)
                .unwrap()
                .iter()
                .find(|(r, _)| *r == row)
                .map_or(0., |(_, value)| *value);
            assert_eq!(actual, 1.);
        }
        let mut pricing = snapshot.pricing.fork_for_node();
        let priced = pricing.price_phase(0, 20, false).unwrap();
        assert_eq!(priced.minimum_reduced_cost, 0.);
        let inputs = &pricing.engine.as_ref().unwrap().cuts.kpath;
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].support, vec![false, true, true, true, false]);
        assert_eq!(
            inputs[0].dual.to_bits(),
            reference["pricing_value"].as_f64().unwrap().to_bits()
        );
    }
}

#[test]
fn restored_fleet_siblings_match_capacity_oracles_without_mutating_parent() {
    for paired in [false, true] {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../fixtures/cuts/metasolver-fleet-restore.json"
        ))
        .unwrap();
        let graph = cut_lifecycle_graph(&fixture, paired);
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        options.cg.artificial_cost = 1000.;
        let parent = super::super::solve(
            RoutingContext::set_partitioning(graph, 3.).unwrap(),
            options.clone(),
        )
        .unwrap();
        assert_eq!(parent.master.lp, 1.875);
        let snapshot = parent.snapshot.as_ref().unwrap();
        // The three pairs at quantity 1/2 attain 3/2 + (.125+.25+.375)/2.
        // At least two vehicles cost 2 plus the cheapest pair edge (.125);
        // one capacity-two vehicle cannot cover three unit demands.
        for sense in [RowSense::GreaterEqual, RowSense::LessEqual] {
            let rhs = if sense == RowSense::GreaterEqual {
                2.
            } else {
                1.
            };
            let child = super::super::inherit_branches(
                snapshot,
                &[BranchConstraint::NumberOfVehicles {
                    graph_id: 7,
                    sense,
                    rhs,
                }],
                options.clone(),
            )
            .unwrap();
            if sense == RowSense::GreaterEqual {
                assert_eq!(child.master.status, cg::Status::Optimal);
                assert_eq!(child.master.lp, 2.125);
            } else {
                assert!(
                    child.preprocessing.infeasible || child.master.status == cg::Status::Infeasible,
                    "capacity-one fleet child must be infeasible: {child:?}"
                );
                if paired && !child.preprocessing.infeasible {
                    assert!(child
                        .master
                        .feasibility
                        .iter()
                        .any(|report| report.certified_infeasible));
                }
            }
            let restored = super::super::inherit_branches(snapshot, &[], options.clone()).unwrap();
            assert_eq!(restored.master.status, cg::Status::Optimal);
            assert_eq!(restored.master.lp, 1.875);
            assert_eq!(snapshot.master.objective, 1.875);
        }
    }
}

#[test]
fn restored_ryan_foster_siblings_match_pair_cover_oracles() {
    use crate::pricing::restrictions::RyanFosterConstraint;
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/cuts/metasolver-fleet-restore.json"
    ))
    .unwrap();
    for paired in [false, true] {
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        options.cg.stabilize = false;
        options.cg.artificial_cost = 1000.;
        let parent = super::super::solve(
            RoutingContext::set_partitioning(cut_lifecycle_graph(&fixture, paired), 3.).unwrap(),
            options.clone(),
        )
        .unwrap();
        let snapshot = parent.snapshot.as_ref().unwrap();
        assert_eq!(parent.master.lp, 1.875);
        // Capacity permits only singletons and pairs. Together forces pair
        // {1,2} plus singleton {3}; apart chooses pair {1,3} plus singleton {2}.
        // Their exact cover costs are 2+.125 and 2+.25, respectively.
        for (together, expected) in [(true, 2.125), (false, 2.25)] {
            let rule = RyanFosterConstraint {
                first: 1,
                second: 2,
                together,
            };
            let child = super::super::inherit_branches(
                snapshot,
                &[BranchConstraint::RyanFoster(rule)],
                options.clone(),
            )
            .unwrap();
            assert_eq!(
                child.master.status,
                cg::Status::Optimal,
                "paired={paired}, together={together}"
            );
            assert_eq!(child.master.lp, expected);
            for (route, value) in child.route_values() {
                if value > 1e-8 {
                    assert!(rule.satisfied_by(route.nodes.iter().map(|&v| v as usize)));
                }
            }
            let restored = super::super::inherit_branches(snapshot, &[], options.clone()).unwrap();
            assert_eq!(restored.master.lp, 1.875);
            assert_eq!(snapshot.master.objective, 1.875);
        }
    }
}
