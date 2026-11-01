use super::repair_theta_duals;
use super::{prefer_arc_memory, separation_gap_closed};
use crate::cuts::{MemoryKind, MemoryType};
use crate::master::model::master_variables;
use crate::master::model::{Options, Problem};

#[test]
fn theta_dual_repair_proves_static_feasibility_without_changing_row_signs() {
    use crate::extensions::stochastic::{Demand, Policy};
    let stochastic = crate::extensions::stochastic::Problem::new(
        2,
        vec![Demand::new(vec![(1, 1.)]).unwrap()],
        vec![vec![0., 1.], vec![1., 0.]],
    )
    .unwrap();
    let (problem, recourse) = stochastic.theta_cut_model(Policy::Classical, 1, 1).unwrap();
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    options.separation.route_recourse = Some(recourse);
    let mut output = super::super::solve(problem, options).unwrap();
    let state = &mut output.snapshot.state;
    let variable = state
        .cuts
        .as_ref()
        .unwrap()
        .theta
        .iter()
        .flatten()
        .next()
        .copied()
        .unwrap();
    let row = state.shared_offset + state.master_variables[variable].entries[0].0;
    let mut duals = vec![0.; state.rhs.len()];
    duals[row] = 1. + f64::EPSILON;
    assert_eq!(
        master_variables::reduced_cost_nonnegative(
            &state.master_variables[variable],
            1.,
            &duals,
            state.shared_offset
        ),
        Some(false)
    );
    repair_theta_duals(state, &mut duals);
    assert!(duals[row] >= 0. && duals[row] <= 1.);
    assert_eq!(
        master_variables::reduced_cost_nonnegative(
            &state.master_variables[variable],
            1.,
            &duals,
            state.shared_offset
        ),
        Some(true)
    );
    // A shared static row requires a joint projection. Decline it instead of
    // changing another variable's dual constraint through a local repair.
    let other = state.master_variables[variable].clone();
    state.master_variables.push(other);
    duals[row] = 1. + f64::EPSILON;
    repair_theta_duals(state, &mut duals);
    assert_eq!(duals[row], 1. + f64::EPSILON);
}

#[test]
fn separation_gap_requires_a_finite_and_close_incumbent() {
    assert!(!separation_gap_closed(100., f64::INFINITY));
    assert!(!separation_gap_closed(100., 101.));
    assert!(separation_gap_closed(100., 100.0000001));
}

fn rc101_networks(customers: usize, copies: i32) -> Problem {
    use crate::inputs::{Instance, Rounding};
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    let mut problem = Problem::set_partitioning(graph, customers as f64).unwrap();
    for id in 1..copies {
        let mut network = problem.networks[0].clone();
        network.id = id;
        // A heterogeneous copy: every arc costs a little more.
        for arc in &mut network.graph.arcs {
            arc.cost *= 1. + 0.05 * id as f64;
        }
        problem.networks.push(network);
    }
    problem
}

#[test]
fn several_networks_share_rank_one_cuts_over_their_packing_rows() {
    let mut plain = Options::default();
    plain.cg.stabilize = false;
    let mut cut = plain.clone();
    cut.separation.rank_one.enabled = true;
    let without = super::super::solve(rc101_networks(25, 2), plain.clone()).unwrap();
    let with = super::super::solve(rc101_networks(25, 2), cut.clone()).unwrap();
    assert!(with.cuts > 0, "{}", with.cuts);
    assert!(with.master.lp >= without.master.lp - 1e-6);
    // Every network prices the same cuts, aligned by registry index.
    let lengths: Vec<_> = with
        .snapshot
        .state
        .oracles
        .iter()
        .map(|o| o.solver.cuts.cuts.len())
        .collect();
    assert!(lengths.windows(2).all(|w| w[0] == w[1]), "{lengths:?}");
    // A single network gives the same row space and keeps its own path.
    let single = super::super::solve(rc101_networks(25, 1), cut.clone()).unwrap();
    assert!(single.cuts > 0);
    // Capacity cuts are defined in the first network's routing layout and
    // mapped to every network through the shared packing rows.
    use crate::inputs::{Instance, Rounding};
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(25).unwrap();
    let mut capacity = plain.clone();
    capacity.separation.capacity = Some(instance.capacity_cut_config(Rounding::None, false));
    let base = super::super::solve(rc101_networks(25, 2), plain).unwrap();
    let both = super::super::solve(rc101_networks(25, 2), capacity).unwrap();
    assert!(both.master.lp >= base.master.lp - 1e-6);
    // Relaxation improvement strengthens each network independently.
    let mut general = cut;
    general.separation.relaxation_improvement_priority = 1.;
    assert!(super::super::solve(rc101_networks(5, 2), general)
        .unwrap()
        .master
        .lp
        .is_finite());
}

#[test]
fn prefer_arc_memory_matches_recovered_score_and_tolerance_gates() {
    // Equal bounds: 2^10 == 1024, so score = 1 / (T_arc / T_node), rejected by strict bound gate.
    let (prefer, score) = prefer_arc_memory(926.25, 926.25, f64::INFINITY, 0.0249599, 0.0707063);
    assert!(!prefer);
    assert!((score - 0.353008).abs() < 1e-5, "{score}");

    // Worse arc bound with finite incumbent (inputs rounded to 6 printed digits).
    let (prefer, score) = prefer_arc_memory(566.875898, 564.957, 567.3, 2.85324, 15.4078);
    assert!(!prefer);
    assert!((score - 4.41707e-15).abs() < 5e-17, "{score}");

    // Substantial bound gain overcoming a 4.25x cumulative runtime ratio.
    let (prefer, score) = prefer_arc_memory(
        519.4714285714286,
        535.1603084415584,
        545.0,
        0.196612,
        0.836278,
    );
    assert!(prefer);
    assert!((score - 16.6438).abs() < 1e-2, "{score}");

    // Strict bound-improvement tolerance: max(|B_node|, |B_arc|) * 1e-10 + 1e-6.
    let (prefer_within_tol, score_within_tol) =
        prefer_arc_memory(500.0, 500.0 + 5e-7, 510.0, 1.0, 0.2);
    assert!(score_within_tol > 1.0, "{score_within_tol}");
    assert!(!prefer_within_tol);
    let (prefer_above_tol, _) = prefer_arc_memory(500.0, 500.0 + 2e-6, 510.0, 1.0, 0.2);
    assert!(prefer_above_tol);

    // Strictly better bound rejected when score <= 1.0.
    let (prefer_slow, score_slow) = prefer_arc_memory(500.0, 501.0, 510.0, 1.0, 5.0);
    assert!(!prefer_slow);
    assert!(score_slow < 1.0, "{score_slow}");

    // Zero completed rank-one rounds in Pass 1 (T_node == 0) yields score 0 and keeps vertex memory.
    assert_eq!(
        prefer_arc_memory(500.0, 505.0, 510.0, 0.0, 0.5),
        (false, 0.0)
    );
    assert_eq!(
        prefer_arc_memory(500.0, 505.0, 510.0, 0.0, 0.0),
        (false, 0.0)
    );
}

#[test]
fn automatic_rank_one_memory_evaluates_root_and_persists_choice_to_children() {
    use crate::inputs::{Instance, Rounding};
    let root_dir = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root_dir}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(25).unwrap();

    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.max_cut_rounds = 4;
    options.separation.capacity = Some(instance.capacity_cut_config(Rounding::None, false));
    options.separation.rank_one.enabled = true;
    options.separation.rank_one.tailing_off_counter = 100;
    options.pricing_options = vec![
        ("BALDESrankOneCutsMemoryType".to_string(), 0.),
        ("BALDESrankOneCutsMaxNumRows".to_string(), 3.),
    ];

    let root = super::super::solve(rc101_networks(25, 1), options.clone()).unwrap();
    let cuts_state = root.snapshot.state.cuts.as_ref().unwrap();
    let resolved = cuts_state
        .resolved_memory_type
        .expect("root automatic search must resolve memory type");
    if cfg!(feature = "rank-one-node-memory") {
        assert_eq!(resolved, MemoryType::Vertex);
    } else {
        assert!(matches!(resolved, MemoryType::Vertex | MemoryType::Arc));
    }
    let expected_kind = match resolved {
        MemoryType::Arc => MemoryKind::Arc,
        _ => MemoryKind::Vertex,
    };
    let registry = &root.snapshot.state.oracles[0].solver.cuts;
    assert!(
        !registry.cuts.is_empty(),
        "expected separated rank-one cuts"
    );
    for (idx, _) in &cuts_state.rows {
        assert_eq!(registry.cuts[*idx].memory_kind, expected_kind);
    }

    // Forking a child node whose options still pass BALDESrankOneCutsMemoryType = 0
    // inherits the root's resolved choice rather than re-running automatic search.
    let mut child_snapshot = root.snapshot.fork().unwrap();
    child_snapshot.state.node_depth = 1;
    let child = child_snapshot.solve(options).unwrap();
    assert_eq!(
        child
            .snapshot
            .state
            .cuts
            .as_ref()
            .and_then(|c| c.resolved_memory_type),
        Some(resolved)
    );
    assert_eq!(
        child.snapshot.state.oracles[0]
            .solver
            .options
            .get("BALDESrankOneCutsMemoryType")
            .unwrap() as i32,
        resolved as i32
    );
}

#[test]
fn automatic_rank_one_memory_skips_arc_pass_on_tailing_off_and_closed_incumbent() {
    let mut tailed = Options::default();
    tailed.cg.stabilize = false;
    tailed.separation.rank_one.enabled = true;
    tailed.separation.rank_one.min_rounds_before_stop_by_sp = 1;
    tailed.separation.rank_one.tailing_off_counter = 1;
    tailed.separation.rank_one.tailing_off_threshold = 100.0;
    tailed.pricing_options = vec![
        ("BALDESrankOneCutsMemoryType".to_string(), 0.),
        ("BALDESrankOneCutsMaxNumRows".to_string(), 3.),
        ("BALDESrankOneCutsMaxNumPerRound".to_string(), 1.),
    ];
    let out = super::super::solve(rc101_networks(25, 1), tailed).unwrap();
    let cuts = out.snapshot.state.cuts.as_ref().unwrap();
    assert!(cuts.stopped_by_tailing_off);
    assert_eq!(cuts.resolved_memory_type, Some(MemoryType::Vertex));

    // When Pass 1 already closes the incumbent gap, it keeps Vertex memory without Pass 2.
    let mut closed = Options::default();
    closed.cg.stabilize = false;
    closed.separation.rank_one.enabled = true;
    closed.separation.upper_bound = Some(out.master.bound);
    closed.pricing_options = vec![("BALDESrankOneCutsMemoryType".to_string(), 0.)];
    let closed_out = super::super::solve(rc101_networks(25, 1), closed).unwrap();
    assert_eq!(
        closed_out
            .snapshot
            .state
            .cuts
            .as_ref()
            .and_then(|c| c.resolved_memory_type),
        Some(MemoryType::Vertex)
    );
}

#[test]
fn every_network_receives_knapsack_resource_mapping_and_exact_fenchel_mode() {
    use super::KnapsackCutConfig;
    use crate::cuts::knapsack::KnapsackRow;
    use std::collections::BTreeMap;
    let mut options = Options::default();
    options.cg.stabilize = false;
    let mut row = KnapsackRow::new(0, 7, 1., 10000);
    row.graph_resources.insert(1, (7, 1.));
    options.separation.knapsack = Some(
        KnapsackCutConfig::new(vec![row])
            .with_resource_indices(BTreeMap::from([(7, 0)]))
            .with_fenchel(Default::default()),
    );
    let output = super::super::solve(rc101_networks(5, 2), options).unwrap();
    for oracle in &output.snapshot.state.oracles {
        assert_eq!(oracle.solver.knapsack_indices().get(&7), Some(&0));
        assert!(oracle.solver.knapsack_exact_coefficients);
    }
}

#[test]
fn general_kpath_accepts_and_validates_each_network_graph() {
    use crate::cuts::kpath::KPathGraph;
    use crate::cuts::rank_one::separation::{prepare_incidence, ArcData, GraphData, VertexData};
    let problem = rc101_networks(5, 2);
    let graphs: Vec<_> = problem
        .networks
        .iter()
        .map(|network| {
            let mut graph = GraphData {
                id: network.id,
                source: network.graph.source as i32,
                sink: network.graph.sink as i32,
                elem_set_count: 5,
                packing_set_count: 5,
                covering_set_count: 5,
                ..Default::default()
            };
            for v in 0..network.graph.vertices.len() {
                let sets = if v > 0 && v <= 5 {
                    vec![(v - 1) as i32]
                } else {
                    vec![]
                };
                graph.vertices.push(VertexData {
                    id: v as i32,
                    elem_sets: sets.clone(),
                    packing_sets: sets.clone(),
                    covering_sets: sets,
                });
            }
            graph.arcs = network
                .graph
                .arcs
                .iter()
                .map(|arc| ArcData {
                    id: arc.id,
                    tail: arc.tail as i32,
                    head: arc.head as i32,
                    ..Default::default()
                })
                .collect();
            KPathGraph {
                incidence: prepare_incidence(&graph),
                graph,
                ..Default::default()
            }
        })
        .collect();
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance =
        crate::inputs::Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(5).unwrap();
    let demands = instance
        .customers
        .iter()
        .skip(1)
        .map(|customer| customer.demand)
        .collect();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.separation.general_kpath = Some(super::GeneralKPathCutConfig::new(
        graphs,
        demands,
        instance.capacity,
        true,
    ));
    let valid = super::super::solve(problem.clone(), options.clone()).unwrap();
    assert!(valid.master.lp.is_finite());
    options.separation.general_kpath.as_mut().unwrap().graphs[1]
        .graph
        .arcs
        .clear();
    let error = super::super::solve(problem, options).unwrap_err();
    assert!(error.message.contains("arc incidence"), "{error}");
}

#[test]
fn strong_kpath_keeps_network_registries_and_master_coefficients_aligned() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance =
        crate::inputs::Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(25).unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.separation.kpath = Some(super::KPathCutConfig::new(
        instance
            .customers
            .iter()
            .map(|customer| customer.demand)
            .collect(),
        instance.capacity,
    ));
    let output = super::super::solve(rc101_networks(25, 2), options).unwrap();
    assert!(output.master.lp.is_finite());
    let state = &output.snapshot.state;
    let first = &state.oracles[0];
    for oracle in &state.oracles {
        assert_eq!(oracle.solver.cuts.cuts.len(), first.solver.cuts.cuts.len());
        assert_eq!(oracle.nonlinear_rows, first.nonlinear_rows);
        assert_eq!(oracle.solver.cuts.retired, first.solver.cuts.retired);
    }
    for column in output.snapshot.warm.columns.iter().flatten() {
        let Ok(tag) = usize::try_from(column.tag) else {
            continue;
        };
        let Some(route) = state.routes.get(tag) else {
            continue;
        };
        let oracle = state
            .oracles
            .iter()
            .find(|oracle| oracle.network.id == route.network)
            .unwrap();
        assert_eq!(
            column.entries,
            super::super::entries(oracle, &route.route, state.shared_offset).unwrap()
        );
    }
}

#[test]
fn rank_one_neighbourhood_cap_defaults_to_sixteen_and_reaches_the_separator() {
    let problem = rc101_networks(5, 1);
    let graph = &problem.networks[0].graph;
    let customers = problem.customers();
    let mut registry = crate::cuts::CutRegistry::default();
    let mut options = crate::core::options::Options::default();
    assert_eq!(
        options.get("BALDESrankOneCutsNeighbourhoodSize").unwrap(),
        16.
    );
    options
        .set("BALDESrankOneCutsNeighbourhoodSize", 7.)
        .unwrap();
    registry
        .separate_default(graph, &options, &customers, &[])
        .unwrap();
    assert_eq!(registry.parameters.neighbourhood_size, 7);
    options
        .set("BALDESrankOneCutsNeighbourhoodSize", 0.)
        .unwrap();
    assert!(registry
        .separate_default(graph, &options, &customers, &[])
        .is_err());
}
