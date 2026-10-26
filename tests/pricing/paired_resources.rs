//! End-to-end contracts for the physical paired base/dependent relation.
use baldes::graph::coupled::{
    CoupledResource, PairedResourceArc, PairedResourceMode, PiecewiseLinearFunction,
};
use baldes::{Graph, Window};
use std::collections::BTreeMap;

fn constant(lo: f64, hi: f64, value: f64) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(vec![[lo, value], [hi, value]]).unwrap()
}
fn resource(mode: PairedResourceMode) -> CoupledResource {
    CoupledResource::PairedPiecewiseLinear {
        initial: constant(0., 1., 10.),
        initial_base: Window::new(0., 0.).unwrap(),
        arcs: (0..4)
            .map(|i| {
                let (lo, hi) = if i % 2 == 0 { (3., 8.) } else { (6., 10.) };
                PairedResourceArc {
                    arrival: PiecewiseLinearFunction::new(vec![[0., 1.], [8., 9.]]).unwrap(),
                    increment: constant(0., 8., [2., 1., 3., 2.][i]),
                    lower: constant(lo, hi, 0.),
                    upper: constant(lo, hi, 100.),
                    waiting: [1, 2, 0, 1][i],
                }
            })
            .collect(),
        mode,
    }
}
fn fixture(mode: PairedResourceMode, scalar: f64) -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    graph.elementary = true;
    graph.vertices[0].elements.clear();
    graph.vertices[3].elements.clear();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        let arc = graph.add_arc(u, v, scalar).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph.set_coupled_resources(&[resource(mode)]).unwrap();
    graph
}

// Independent control optimization. Arrival is t+1, head waiting is free to
// choose within the explicitly listed windows. All affine coefficients and
// window endpoints are integer, so the two-control polygon has integer
// vertices; enumerating these controls includes each continuous optimum.
// No production function/front evaluation or resource replay is used.
fn control_oracle(path: &[usize], mode: PairedResourceMode) -> f64 {
    if mode == PairedResourceMode::Feasibility {
        return 0.;
    }
    let increments = [2., 1., 3., 2.];
    let waiting = [1., 2., 0., 1.];
    (3..=8)
        .flat_map(|first| {
            ((first + 1).max(6)..=10).map(move |second| {
                increments[path[0]]
                    + waiting[path[0]] * f64::from(first - 1)
                    + increments[path[1]]
                    + waiting[path[1]] * f64::from(second - first - 1)
            })
        })
        .reduce(f64::min)
        .unwrap()
}
fn expected(graph: &Graph<2>, mode: PairedResourceMode, cut: f64) -> BTreeMap<Vec<i32>, f64> {
    [vec![0usize, 1], vec![2, 3]]
        .into_iter()
        .map(|path| {
            let cost = control_oracle(&path, mode)
                + path
                    .iter()
                    .map(|&i| graph.arcs[i].reduced_cost)
                    .sum::<f64>()
                + cut;
            (path.into_iter().map(|i| graph.arcs[i].id).collect(), cost)
        })
        .collect()
}
fn check_routes(routes: Vec<baldes::Route>, expected: &BTreeMap<Vec<i32>, f64>) {
    assert_eq!(routes.len(), expected.len());
    for route in routes {
        assert!(
            (route.cost - expected[&route.arcs]).abs() < 1e-9,
            "{route:?}"
        );
    }
}

#[test]
fn paired_pricing_cut_duals_fixing_retained_enumeration_and_siblings_match_control_oracle() {
    for mode in [
        PairedResourceMode::CostBounded,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::Feasibility,
    ] {
        for direction in [0., 1., 2.] {
            let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = fixture(mode, -10.);
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            solver
                .cuts
                .add_strong_k_path_cut(4, &[1, 2], 1, -1.25)
                .unwrap();
            let oracle = expected(&solver.graph, mode, 1.25);
            let priced = solver.price_phase(0, 100, false).unwrap();
            assert!(priced.exact);
            assert!(
                (priced.minimum_reduced_cost - oracle.values().copied().reduce(f64::min).unwrap())
                    .abs()
                    < 1e-9
            );
            for column in &priced.columns {
                assert!((column.original_cost - (oracle[&column.arcs] - 1.25)).abs() < 1e-9);
            }
            check_routes(solver.enumerate(100.).unwrap(), &oracle);
            let mut parent = solver.fork_for_node();
            for (i, arc) in solver.graph.arcs.iter_mut().enumerate() {
                arc.reduced_cost += (i % 2) as f64;
            }
            let refreshed = expected(&solver.graph, mode, 1.25);
            check_routes(
                solver.inspect_enumeration(100.).unwrap().unwrap(),
                &refreshed,
            );
            solver.fix(-12.).unwrap();
            let kept = refreshed.into_iter().filter(|(_, c)| *c < -12.).collect();
            check_routes(solver.enumerate(-12.).unwrap(), &kept);
            check_routes(parent.inspect_enumeration(100.).unwrap().unwrap(), &oracle);
            let mut child = parent.fork_for_node();
            child.set_forbidden_arcs(vec![0]).unwrap();
            let child_expected = oracle
                .iter()
                .filter(|(p, _)| !p.contains(&0))
                .map(|(p, c)| (p.clone(), *c))
                .collect();
            check_routes(child.enumerate(100.).unwrap(), &child_expected);
            check_routes(parent.inspect_enumeration(100.).unwrap().unwrap(), &oracle);
        }
    }
}

#[test]
fn paired_limited_memory_cut_dual_refresh_keeps_original_route_costs() {
    use baldes::cuts::state::LimitedInput;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    // Both customer vertices contribute one half to this cut, so the two
    // one-customer routes have floor(1/2)=0. An arc coefficient supplies the
    // second half and makes each complete route's cut coefficient one.
    let graph = fixture(PairedResourceMode::CostBounded, -10.);
    let cut = |dual| LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual,
        vertex_coefficient: vec![0, 1, 1, 0],
        arc_coefficient: [(graph.arcs[1].id, 1), (graph.arcs[3].id, 1)]
            .into_iter()
            .collect(),
        memory_vertices: (0..4).collect(),
        memory_arcs: Default::default(),
    };
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                bidirectional,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        assert!(engine.load_cuts(&graph, vec![cut(-2.)], vec![]));
        for dual in [-2., -7.] {
            engine.update_cut_duals(vec![cut(dual)], vec![]);
            let priced = engine.solve(&graph, &Default::default()).unwrap();
            assert!((priced.minimum_reduced_cost - (-15. + dual)).abs() < 1e-9);
            for column in priced.columns {
                let original = if column.arcs.contains(&graph.arcs[0].id) {
                    -13.
                } else {
                    -15.
                };
                assert!((column.original_cost - original).abs() < 1e-9);
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn paired_phase_i_restores_costs_and_certified_controls_allow_complete_transfer() {
    use baldes::master::{
        cg::{FeasibilityStatus, Status},
        model::{self, format::Model, EnumerationOptions, Options},
        routing::{self, RootOptions, RoutingContext},
    };
    let graph = fixture(PairedResourceMode::CostBounded, 0.);
    let mut options = RootOptions {
        enumerate: false,
        round_reduced_costs: false,
        ..Default::default()
    };
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    let output = routing::solve(
        RoutingContext::set_partitioning(graph, 2.).unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(output.master.status, Status::Optimal, "{output:?}");
    assert!((output.master.lp - 12.).abs() < 1e-8);
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|r| r.status == FeasibilityStatus::Feasible));
    assert!(
        (output
            .route_values()
            .iter()
            .map(|(r, x)| r.original_cost * x)
            .sum::<f64>()
            - 12.)
            .abs()
            < 1e-8
    );

    let mut model: serde_json::Value =
        serde_json::from_str(include_str!("../../examples/networks/interval_cost.json")).unwrap();
    model["networks"][0]["coupled_resources"] =
        serde_json::json!([resource(PairedResourceMode::CostBounded)]);
    let problem = Model::from_json(&model.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.artificial_cost = 100.;
    options.cg.stabilize = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    assert!((parent.master.lp - 12.).abs() < 1e-8);
    let attempted = parent
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    let report = attempted.enumeration_report();
    assert!(report.complete, "{report:?}");
    assert_eq!(report.routes, 2);
    assert!(report.failure.is_none());
    assert!(!parent.snapshot.enumeration_report().complete);
    let resumed = attempted.solve(options).unwrap();
    assert_eq!(resumed.master.status, Status::Optimal);
    assert!((resumed.master.lp - parent.master.lp).abs() < 1e-8);
}

#[cfg(feature = "clp")]
#[test]
fn paired_json_round_trip_keeps_modes_functions_waiting_and_domains() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        let value = resource(mode);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("paired_piecewise_linear"));
        assert_eq!(
            serde_json::from_str::<CoupledResource>(&encoded).unwrap(),
            value
        );
        let mut malformed: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        malformed["arcs"][0]["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CoupledResource>(malformed).is_err());
    }
}
