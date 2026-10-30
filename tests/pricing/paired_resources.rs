//! End-to-end contracts for the physical paired base/dependent relation.
use baldes::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
};
use baldes::graph::coupled::{CoupledResource, MachineArcFunctionInput, PairedResourceMode};
use baldes::{Graph, Window};
use std::collections::BTreeMap;

const INCREMENTS: [f64; 4] = [2., 1., 3., 2.];
const WAITING: [i32; 4] = [1, 2, 0, 1];
// Base windows: source, the two customers, sink.
const WINDOWS: [(f64, f64); 4] = [(0., 1.), (3., 8.), (3., 8.), (6., 10.)];
const ENDPOINTS: [(usize, usize); 4] = [(0, 1), (1, 3), (0, 2), (2, 3)];

fn resource(mode: PairedResourceMode) -> CoupledResource {
    CoupledResource::PairedFront { mode }
}
fn constant(window: (f64, f64), value: f64) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(vec![(value, window.0), (0., window.1)]).unwrap()
}
// Unit base consumption and constant dependent increments, with function
// domains equal to the tail windows as in the native program.
fn install_program(graph: &mut Graph<2>, resource: usize) {
    let window = |(lower, upper)| Window::new(lower, upper).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let base: Vec<_> = ENDPOINTS
        .iter()
        .map(|&(tail, _)| constant(WINDOWS[tail], 1.))
        .collect();
    let inputs: Vec<_> = ENDPOINTS
        .iter()
        .enumerate()
        .map(|(arc, &(tail, head))| MachineArcFunctionInput {
            physical_arc: arc,
            tail_bounds: window(WINDOWS[tail]),
            waiting_bounds: window(WINDOWS[head]),
            waiting_rate: WAITING[arc],
            base: &base[arc],
            dependent: affine(INCREMENTS[arc]),
            dependent_lower: affine(0.),
            dependent_upper: affine(100.),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        resource,
        &inputs,
        affine(0.),
        window(WINDOWS[0]),
        window(WINDOWS[3]),
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
}
#[cfg(feature = "clp")]
fn program_json(resource: usize) -> serde_json::Value {
    let constant = |window: (f64, f64), value: f64| serde_json::json!({"pairs": [[value, window.0], [0., window.1]]});
    let affine =
        |value: f64| serde_json::json!({"encoding": "affine", "intercept": value, "slope": 0.});
    let arcs: Vec<_> = ENDPOINTS
        .iter()
        .enumerate()
        .map(|(arc, &(tail, head))| {
            serde_json::json!({
                "tail_bounds": [WINDOWS[tail].0, WINDOWS[tail].1],
                "waiting_bounds": [WINDOWS[head].0, WINDOWS[head].1],
                "waiting_rate": f64::from(WAITING[arc]),
                "base": constant(WINDOWS[tail], 1.),
                "dependent": affine(INCREMENTS[arc]),
                "dependent_lower": affine(0.),
                "dependent_upper": affine(100.),
            })
        })
        .collect();
    serde_json::json!({
        "resource": resource,
        "source_bounds": [WINDOWS[0].0, WINDOWS[0].1],
        "terminal_bounds": [WINDOWS[3].0, WINDOWS[3].1],
        "root": affine(0.),
        "arcs": arcs,
    })
}
fn fixture(mode: PairedResourceMode, scalar: f64) -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    graph.elementary = true;
    graph.vertices[0].elements.clear();
    graph.vertices[3].elements.clear();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v) in ENDPOINTS {
        let arc = graph.add_arc(u, v, scalar).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph.set_coupled_resources(&[resource(mode)]).unwrap();
    install_program(&mut graph, 0);
    graph
}

// Independent control optimization. Arrival is departure plus one, and head
// waiting is free to choose within the listed windows. The latest source
// departure is 1. All coefficients and window endpoints are integer, so the
// two-control polygon has integer vertices; enumerating these controls
// includes each continuous optimum. On both paths the second waiting rate is
// at least the first, where every native join split attains this optimum.
// No production function/front evaluation or resource replay is used.
fn control_oracle(path: &[usize], mode: PairedResourceMode) -> f64 {
    if mode == PairedResourceMode::Feasibility {
        return 0.;
    }
    let (increments, waiting) = (INCREMENTS, WAITING.map(f64::from));
    (3..=8)
        .flat_map(|first| {
            ((first + 1).max(6)..=10).map(move |second| {
                increments[path[0]]
                    + waiting[path[0]] * f64::from(first - 2)
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
                    -14.
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
    // Control-oracle optimum: 6 on [0, 1] plus 5 on [2, 3].
    assert!((output.master.lp - 11.).abs() < 1e-8);
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
            - 11.)
            .abs()
            < 1e-8
    );

    let mut model: serde_json::Value =
        serde_json::from_str(include_str!("../../examples/networks/interval_cost.json")).unwrap();
    model["networks"][0]["coupled_resources"] =
        serde_json::json!([resource(PairedResourceMode::CostBounded)]);
    model["networks"][0]["paired_front_programs"] = serde_json::json!([program_json(0)]);
    let problem = Model::from_json(&model.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = Options::default();
    options.cg.artificial_cost = 100.;
    options.cg.stabilize = false;
    let parent = model::solve(problem, options.clone()).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    assert!((parent.master.lp - 11.).abs() < 1e-8);
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
fn paired_json_round_trip_keeps_modes_and_front_programs() {
    use baldes::master::model::format::{Model, PairedFrontProgram};
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        let value = resource(mode);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("paired_front"));
        assert_eq!(
            serde_json::from_str::<CoupledResource>(&encoded).unwrap(),
            value
        );
        let mut malformed: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        malformed["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CoupledResource>(malformed).is_err());
    }
    let program: PairedFrontProgram = serde_json::from_value(program_json(0)).unwrap();
    let encoded = serde_json::to_value(&program).unwrap();
    assert_eq!(encoded, program_json(0));
    let mut malformed = program_json(0);
    malformed["arcs"][0]["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<PairedFrontProgram>(malformed).is_err());
    // A model-installed program prices like the directly prepared one.
    let mut model: serde_json::Value =
        serde_json::from_str(include_str!("../../examples/networks/interval_cost.json")).unwrap();
    model["networks"][0]["coupled_resources"] =
        serde_json::json!([resource(PairedResourceMode::CostBounded)]);
    model["networks"][0]["paired_front_programs"] = serde_json::json!([program_json(0)]);
    let problem = Model::from_json(&model.to_string())
        .unwrap()
        .problem::<2>()
        .unwrap();
    let kernel = problem.networks[0].graph.coupled_kernel().unwrap();
    for path in [[0, 1], [2, 3]] {
        assert_eq!(
            kernel.replay(&path).unwrap(),
            Some(control_oracle(&path, PairedResourceMode::CostBounded))
        );
    }
    let mut duplicated = model.clone();
    duplicated["networks"][0]["paired_front_programs"] =
        serde_json::json!([program_json(0), program_json(0)]);
    assert!(Model::from_json(&duplicated.to_string())
        .unwrap()
        .problem::<2>()
        .is_err());
}

#[test]
fn paired_standard_resources_cost_refresh_and_node_forks_keep_their_domains() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        for direction in [0., 1., 2.] {
            let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = fixture(mode, -10.);
            for (index, arc) in solver.graph.arcs.iter_mut().enumerate() {
                arc.variables = vec![(index, 1.)];
            }
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            solver.enumeration_preserve_arc_paths = true;
            for prices in [[0.; 4], [-20., 0., 0., 0.], [0., 0., -20., 0.]] {
                solver.cache_variable_costs(&prices).unwrap();
                let expected: BTreeMap<_, _> = [[0, 1], [2, 3]]
                    .into_iter()
                    .map(|path| {
                        (
                            path.into_iter().map(|a| a as i32).collect(),
                            control_oracle(&path, mode) - 20.
                                + path.into_iter().map(|a| prices[a]).sum::<f64>(),
                        )
                    })
                    .collect();
                let priced = solver.price_functor_phase(0, 20).unwrap();
                assert!(priced.exact);
                assert_eq!(
                    priced.minimum_reduced_cost,
                    expected.values().copied().reduce(f64::min).unwrap()
                );
                check_routes(solver.enumerate(100.).unwrap(), &expected);
                let mut child = solver.fork_for_node();
                child.set_forbidden_arcs(vec![0]).unwrap();
                let child_expected = BTreeMap::from([(vec![2, 3], expected[&vec![2, 3]])]);
                check_routes(child.enumerate(100.).unwrap(), &child_expected);
                check_routes(
                    solver.inspect_enumeration(100.).unwrap().unwrap(),
                    &expected,
                );
                let before: Vec<_> = solver.graph.arcs.iter().map(|a| a.reduced_cost).collect();
                for bad in [vec![f64::NAN; 4], vec![1.; 3]] {
                    assert!(solver.cache_variable_costs(&bad).is_err());
                    assert_eq!(
                        solver
                            .graph
                            .arcs
                            .iter()
                            .map(|a| a.reduced_cost)
                            .collect::<Vec<_>>(),
                        before
                    );
                }
            }
        }
    }
}

#[test]
fn paired_owner_coefficients_keep_raw_pricing_and_selected_original_values_distinct() {
    for direction in [0., 1., 2.] {
        for coefficient in [-2., -1., 0., 0.5, 1., 2.] {
            let mode = PairedResourceMode::CostBounded;
            let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = fixture(mode, -10.);
            solver
                .graph
                .set_coupled_objective_coefficients(&[(0, coefficient)])
                .unwrap();
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            solver.enumeration_preserve_arc_paths = true;
            let expected: BTreeMap<_, _> = [[0, 1], [2, 3]]
                .into_iter()
                .map(|path| {
                    let value = control_oracle(&path, mode);
                    let kernel = solver.graph.coupled_kernel().unwrap();
                    assert_eq!(kernel.replay_pricing(&path).unwrap(), Some(value));
                    assert_eq!(kernel.replay(&path).unwrap(), Some(coefficient * value));
                    let components = kernel.replay_cost_components(&path).unwrap().unwrap();
                    assert_eq!(components.total, coefficient * value);
                    assert_eq!(components.resources, vec![coefficient * value]);
                    (path.into_iter().map(|a| a as i32).collect(), value - 20.)
                })
                .collect();
            let priced = solver.price_phase(0, 20, false).unwrap();
            assert_eq!(
                priced.minimum_reduced_cost,
                expected.values().copied().reduce(f64::min).unwrap()
            );
            for route in &priced.columns {
                let path: Vec<_> = route.arcs.iter().map(|&a| a as usize).collect();
                assert_eq!(
                    route.original_cost,
                    -20. + coefficient * control_oracle(&path, mode)
                );
                assert_eq!(route.coupled_values, vec![(0, control_oracle(&path, mode))]);
            }
            check_routes(solver.enumerate(100.).unwrap(), &expected);
        }
    }
}

#[derive(Debug)]
struct UnsupportedPlugin;
impl baldes::graph::coupled::CoupledPlugin for UnsupportedPlugin {
    fn has_costs(&self) -> bool {
        true
    }
    fn validate(&self, _: usize) -> baldes::Result<()> {
        Ok(())
    }
    fn root(&self, _: bool) -> baldes::graph::coupled::PluginState {
        baldes::graph::coupled::PluginState::new(())
    }
    fn extend(
        &self,
        state: &baldes::graph::coupled::PluginState,
        _: usize,
        _: bool,
    ) -> baldes::Result<Option<baldes::graph::coupled::PluginState>> {
        Ok(Some(state.clone()))
    }
    fn join(
        &self,
        _: &baldes::graph::coupled::PluginState,
        _: &mut dyn Iterator<Item = usize>,
    ) -> baldes::Result<bool> {
        Ok(true)
    }
}

#[test]
fn paired_factory_limits_reject_other_coupled_compositions_without_replacing_the_installed_program()
{
    use baldes::graph::coupled::{FrontOperationPolicy, PiecewiseLinearFunction, PluginResource};
    let bounds = Window::new(0., 10.).unwrap();
    let extras = vec![
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1.; 4],
            weights: vec![1.; 4],
            sink_upper: None,
            source_upper: None,
        },
        CoupledResource::CustomResource {
            increments: vec![1.; 4],
            weights: vec![1.; 4],
            forward_bound: None,
            backward_bound: None,
        },
        CoupledResource::Selection {
            initial: 1,
            allowed: vec![1; 4],
            vertices: vec![],
            relaxed: false,
        },
        CoupledResource::SimultaneousPickupDelivery {
            capacity: 10.,
            deliveries: vec![1.; 4],
            pickups: vec![0.; 4],
            tolerance: 0.,
        },
        CoupledResource::BudgetedLoad {
            capacity: 10.,
            budget: 1,
            demands: vec![1.; 4],
            deviations: vec![0.; 4],
        },
        CoupledResource::BudgetedTime {
            initial: 0.,
            budget: 1,
            durations: vec![1.; 4],
            deviations: vec![0.; 4],
            windows: vec![bounds; 4],
        },
        CoupledResource::IntervalCost {
            initial: bounds,
            bounds,
            increments: vec![0.; 4],
            functions: vec![PiecewiseLinearFunction::new(vec![[0., 1.], [10., 1.]]).unwrap(); 4],
        },
        CoupledResource::SlopeIntervalCost {
            initial: bounds,
            bounds,
            increments: vec![0.; 4],
            functions: vec![constant((0., 10.), 1.); 4],
            policy: FrontOperationPolicy::StrictDomain,
        },
        CoupledResource::Plugin(PluginResource::new(UnsupportedPlugin)),
    ];
    let mode = PairedResourceMode::CostUnbounded;
    let mut graph = fixture(mode, 0.);
    for extra in extras {
        let error = graph
            .set_coupled_resources(&[resource(mode), extra])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported paired resource combination"),
            "{error}"
        );
        assert_eq!(
            graph.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
            Some(control_oracle(&[0, 1], mode))
        );
    }
    let error = graph
        .set_coupled_resources(&[resource(mode), resource(mode)])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("only one paired resource program"),
        "{error}"
    );
    assert_eq!(
        graph.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
        Some(control_oracle(&[0, 1], mode))
    );
}
