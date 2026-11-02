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

#[test]
fn changed_rank_one_memory_refreshes_installed_coefficients_in_each_memory_mode() {
    use crate::cuts::{CutClass, RankOneCut};
    use crate::graph::Graph;
    use crate::master::{LpMaster, RowSense};
    use std::collections::{BTreeMap, BTreeSet};
    for memory_kind in [MemoryKind::Vertex, MemoryKind::Arc] {
        for dependent in [false, true] {
            let mut graph = Graph::<2>::new(5, 0).unwrap();
            for v in 0..4 {
                graph.add_arc(v, v + 1, 1.).unwrap();
            }
            let problem = Problem::set_partitioning(graph, 3.).unwrap();
            let mut options = Options::default();
            options.separation.rank_one.enabled = true;
            let mut output = super::super::solve(problem, options).unwrap();
            let snapshot = &mut output.snapshot;
            let tagged: Vec<_> = snapshot
                .warm
                .columns
                .iter()
                .enumerate()
                .filter_map(|(column, c)| {
                    c.as_ref()
                        .filter(|c| c.tag >= 0)
                        .map(|c| (column, c.tag as usize))
                })
                .collect();
            let column = tagged
                .iter()
                .find(|(_, tag)| snapshot.state.routes[*tag].route.nodes == [0, 1, 2, 3, 4])
                .unwrap()
                .0;
            let memory = vec![false, true, false, true, false];
            let arcs = BTreeSet::from([1]);
            let registry = &mut snapshot.state.oracles[0].solver.cuts;
            assert!(registry.cuts.is_empty());
            registry.cuts.push(RankOneCut {
                weights: vec![0, 1, 0, 1, 0],
                graph_support: BTreeMap::new(),
                graph_pricing_support: BTreeMap::new(),
                general_kpath_support: None,
                memory: memory.clone(),
                graph_dependent_memory: dependent,
                graph_memories: if dependent {
                    BTreeMap::from([(0, memory)])
                } else {
                    BTreeMap::new()
                },
                memory_kind,
                arc_memory: arcs.clone(),
                graph_arc_memories: if dependent {
                    BTreeMap::from([(0, arcs)])
                } else {
                    BTreeMap::new()
                },
                arc_weights: BTreeMap::new(),
                graph_arc_weights: Default::default(),
                visit_once: false,
                type2: false,
                denominator: 2,
                dual: 0.,
                class: CutClass::Packing,
                rhs: 1,
                kind: 0,
                id: 0,
                ordered_support: Some(vec![(1, 1), (3, 1)]),
            });
            let mut cuts = snapshot.state.cuts.take().unwrap();
            let values = snapshot.master.values().to_vec();
            let row = snapshot.master.rhs().len();
            let mut round = super::Round {
                state: &mut snapshot.state,
                cuts: &mut cuts,
                tagged,
                values,
            };
            let mut rows = round.new_rows(&snapshot.master).unwrap();
            round
                .install_registry_cuts(
                    &mut snapshot.master,
                    &mut rows,
                    0,
                    |_, _| RowSense::LessEqual,
                    false,
                )
                .unwrap();
            round.install(&mut snapshot.master, rows).unwrap();
            let coefficient = |master: &crate::master::clp::ClpMaster| {
                master
                    .column_entries(column)
                    .unwrap()
                    .into_iter()
                    .find(|(r, _)| *r == row)
                    .map_or(0., |(_, v)| v)
            };
            assert_eq!(coefficient(&snapshot.master), 0.);
            let cut = &mut round.state.oracles[0].solver.cuts.cuts[0];
            match (memory_kind, dependent) {
                (MemoryKind::Vertex, false) => cut.memory[2] = true,
                (MemoryKind::Vertex, true) => cut.graph_memories.get_mut(&0).unwrap()[2] = true,
                (MemoryKind::Arc, false) => {
                    cut.arc_memory.insert(2);
                }
                (MemoryKind::Arc, true) => {
                    cut.graph_arc_memories.get_mut(&0).unwrap().insert(2);
                }
            }
            let mut rows = round.new_rows(&snapshot.master).unwrap();
            let changed = round
                .install_registry_cuts(
                    &mut snapshot.master,
                    &mut rows,
                    1,
                    |_, _| RowSense::LessEqual,
                    false,
                )
                .unwrap();
            assert!(
                changed,
                "memory kind={memory_kind:?}, graph dependent={dependent}"
            );
            assert_eq!(
                coefficient(&snapshot.master),
                1.,
                "memory kind={memory_kind:?}, graph dependent={dependent}"
            );
        }
    }
}

#[test]
fn partition_rows_reject_extra_sink_or_route_contributions_to_covered_rows() {
    use crate::master::model::OracleNetwork;
    let mut problem = rc101_networks(5, 1);
    let mut network = problem.networks.remove(0);
    let mut rows = problem.rows;
    let original: OracleNetwork<_> = network.clone().into();
    let covered = super::partition_rows(&rows, &original).unwrap();
    let row = covered[0].0;
    let sink_arc = network
        .graph
        .arcs
        .iter()
        .position(|arc| arc.head == network.graph.sink)
        .unwrap();
    network.arc_entries[sink_arc].push((row, 1.));
    assert!(super::partition_rows(&rows, &network.clone().into()).is_none());
    network.arc_entries[sink_arc].pop();
    network.route_entries.push((row, 1.));
    assert!(super::partition_rows(&rows, &network.clone().into()).is_none());
    network.route_entries.clear();
    let unrelated = rows.len();
    rows.push(rows[0].clone());
    network.arc_entries[sink_arc].push((unrelated, 1.));
    network.route_entries.push((unrelated, 1.));
    assert_eq!(super::partition_rows(&rows, &network.into()), Some(covered));
}

#[test]
fn vertex_cut_rows_reject_static_and_resource_activity_contributions() {
    use crate::master::model::{
        solve_with_master_variables, AccumulatedActivityEntry, MasterVariable,
    };
    use crate::master::preprocessing::Bounds;
    let problem = rc101_networks(5, 1);
    let mut options = Options::default();
    options.separation.rank_one.enabled = true;
    options.preprocessing.enabled = false;
    options.cg.time_limit = std::time::Duration::ZERO;
    let variable = MasterVariable {
        id: 100,
        cost: 0.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: false,
        entries: vec![(0, 1.)],
    };
    let error = solve_with_master_variables(problem.clone(), vec![variable], options.clone())
        .err()
        .expect("a static contribution changes the covered row");
    assert!(
        error.to_string().contains("non-vertex contributions"),
        "{error}"
    );

    let mut unrelated_problem = problem.clone();
    let unrelated_row = unrelated_problem.rows.len();
    unrelated_problem
        .rows
        .push(crate::master::formulation::Row {
            sense: crate::master::RowSense::LessEqual,
            rhs: 1.,
        });
    let unrelated = MasterVariable {
        id: 101,
        cost: 0.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: false,
        entries: vec![(unrelated_row, 1.)],
    };
    assert!(
        solve_with_master_variables(unrelated_problem, vec![unrelated], options.clone()).is_ok()
    );

    let mut activity_problem = problem;
    let graph = &mut activity_problem.networks[0].graph;
    let arc_count = graph.arcs.len();
    graph
        .set_coupled_resources(&[crate::graph::coupled::CoupledResource::CustomResource {
            increments: vec![1.; arc_count],
            weights: vec![1.; arc_count],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    activity_problem.networks[0]
        .activity_entries
        .push(AccumulatedActivityEntry {
            resource: 0,
            row: 0,
            coefficient: 1.,
        });
    let error = super::super::solve(activity_problem, options)
        .err()
        .expect("resource activity changes the covered row");
    assert!(
        error.to_string().contains("non-vertex contributions"),
        "{error}"
    );
}

fn mixed_arc_problem_and_config() -> (Problem<2>, crate::master::model::RankOneIncidenceConfig) {
    use crate::cuts::separation::{ArcData, GraphData, Parameters, VertexData};
    use crate::graph::{Graph, Window};
    use crate::master::formulation::Row;
    use crate::master::model::{Network, RankOneIncidenceConfig};
    use crate::master::RowSense;
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/rank_one_two_graph_mixed_complete_separation_native.json"
    ))
    .unwrap();
    let number = |v: &serde_json::Value| v.as_i64().unwrap() as i32;
    let mut networks = vec![];
    let mut graphs = vec![];
    for (id, arc_key, entry_key) in [
        (7, "arcs7", "master_entries7"),
        (8, "arcs8", "master_entries8"),
    ] {
        let mut graph = Graph::<2>::new(8, 1).unwrap();
        graph.graph_id = id;
        graph.elementary = true;
        let membership = (0..8)
            .map(|v| match v {
                1 => vec![1],
                3..=6 => vec![v - 1],
                _ => vec![],
            })
            .collect();
        graph
            .configure_ng(6, membership, vec![(0..6).collect(); 8])
            .unwrap();
        for v in 0..8 {
            graph.set_bounds(v, 0, 0., 2.).unwrap();
        }
        let mut graph_data = GraphData {
            id,
            source: 0,
            sink: 7,
            elem_set_count: 6,
            packing_set_count: 6,
            covering_set_count: 6,
            vertices: (0..8)
                .map(|v| {
                    let sets = match v {
                        1 => vec![1],
                        3..=6 => vec![v - 1],
                        _ => vec![],
                    };
                    VertexData {
                        id: v,
                        elem_sets: sets.clone(),
                        packing_sets: sets.clone(),
                        covering_sets: sets,
                    }
                })
                .collect(),
            ..Default::default()
        };
        for arc in capture[arc_key].as_array().unwrap() {
            let (arc_id, tail, head) = (
                number(&arc["id"]),
                number(&arc["tail"]) as usize,
                number(&arc["head"]) as usize,
            );
            let cost = if tail == 0 { 1. } else { 0. } + if id == 8 { 0.01 } else { 0. };
            graph.add_arc_with_id(arc_id, tail, head, cost).unwrap();
            let physical = graph.arc_mut(arc_id).unwrap();
            physical.consumption[0] = if head == 7 { 0. } else { 1. };
            let sets = if head == 1 && tail != 4 && tail != 5 {
                vec![0]
            } else {
                vec![]
            };
            physical.elements = sets.iter().map(|&x| x as usize).collect();
            physical.memory = (0..6).collect();
            graph_data.arcs.push(ArcData {
                id: arc_id,
                tail: tail as i32,
                head: head as i32,
                elem_sets: sets.clone(),
                packing_sets: sets.clone(),
                covering_sets: sets,
                ..Default::default()
            });
        }
        let arc_entries = capture[entry_key]
            .as_array()
            .unwrap()
            .iter()
            .map(|entries| {
                entries
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|entry| (number(&entry[0]) as usize, entry[1].as_f64().unwrap()))
                    .collect()
            })
            .collect();
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(0., 6.).unwrap(),
            arc_entries,
            route_entries: vec![],
            activity_entries: vec![],
        });
        graphs.push(graph_data);
    }
    let rows = (0..6)
        .map(|_| Row {
            sense: RowSense::Equal,
            rhs: 1.,
        })
        .collect();
    let parameters = Parameters {
        max_num_rows: 5,
        max_num_per_round: 20,
        memory_type: 1,
        arc_memory_type: 2,
        sp_dependent_memory: false,
        ls_num_iterations: 1000,
        neighbourhood_size: 16,
        max_num_triplets: 10000,
        violation_tolerance: 0.02,
        ..Default::default()
    };
    (
        Problem { rows, networks },
        RankOneIncidenceConfig {
            rows: (0..6).collect(),
            graphs,
            parameters,
        },
    )
}

#[test]
fn explicit_arc_incidence_rank_one_closes_captured_two_network_root() {
    let (problem, config) = mixed_arc_problem_and_config();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options
        .pricing_options
        .push(("BALDESuseBidirectionalSearch".into(), 0.));
    let plain = super::super::solve(problem.clone(), options.clone()).unwrap();
    assert!((plain.master.lp - 2.5).abs() < 1e-7, "{}", plain.master.lp);
    options.separation.rank_one_incidence = Some(config);
    let cut = super::super::solve(problem.clone(), options).unwrap();
    assert!(cut.cuts > 0);
    assert!((cut.master.lp - 3.).abs() < 1e-7, "{}", cut.master.lp);

    // Enumerate every elementary route with at most two customer visits.
    // For integer partitions, no selected row can occur twice on one route.
    let registry = &cut.snapshot.state.oracles[0].solver.cuts;
    let mut candidates = vec![];
    for network in &problem.networks {
        for first in 1..=6 {
            for second in 0..=6 {
                if second == first {
                    continue;
                }
                let mut vertices = vec![0, first];
                if second != 0 {
                    vertices.push(second);
                }
                vertices.push(7);
                let mut path = vec![];
                let mut counts = [0u8; 6];
                for pair in vertices.windows(2) {
                    let (index, arc) = network
                        .graph
                        .arcs
                        .iter()
                        .enumerate()
                        .find(|(_, arc)| arc.tail == pair[0] && arc.head == pair[1])
                        .unwrap();
                    path.push((arc.tail, arc.head, arc.id));
                    for &(row, value) in &network.arc_entries[index] {
                        assert_eq!(value, 1.);
                        counts[row] += 1;
                    }
                }
                if counts.iter().any(|&count| count > 1) {
                    continue;
                }
                let mask = counts
                    .iter()
                    .enumerate()
                    .fold(0u8, |mask, (row, &count)| mask | (count << row));
                let coefficients = registry
                    .cuts
                    .iter()
                    .map(|row| {
                        [
                            crate::cuts::coefficient(row, network.id, &path, false).unwrap(),
                            crate::cuts::coefficient(row, network.id, &path, true).unwrap(),
                        ]
                    })
                    .collect::<Vec<_>>();
                if mask == 0 {
                    assert!(coefficients.iter().all(|&values| values == [0, 0]));
                    continue;
                }
                for (cut, &[limited, full]) in registry.cuts.iter().zip(&coefficients) {
                    assert!(match cut.class {
                        crate::cuts::CutClass::Packing => full >= limited,
                        crate::cuts::CutClass::Covering => full <= limited,
                    });
                }
                candidates.push((mask, coefficients));
            }
        }
    }
    fn partitions(
        mask: u8,
        candidates: &[(u8, Vec<[i32; 2]>)],
        cuts: &[crate::cuts::RankOneCut],
        sums: &mut [[i32; 2]],
        checked: &mut usize,
    ) {
        if mask == 0b11_1111 {
            for (cut, sum) in cuts.iter().zip(sums.iter()) {
                for &value in sum {
                    assert!(
                        match cut.class {
                            crate::cuts::CutClass::Packing => value <= cut.rhs,
                            crate::cuts::CutClass::Covering => value >= cut.rhs,
                        },
                        "cut {cut:?}, value {value}"
                    );
                }
            }
            *checked += 1;
            return;
        }
        let next = (!mask & 0b11_1111).trailing_zeros();
        for (covered, coefficients) in candidates {
            if covered & (1 << next) == 0 || covered & mask != 0 {
                continue;
            }
            for (sum, coefficient) in sums.iter_mut().zip(coefficients) {
                sum[0] += coefficient[0];
                sum[1] += coefficient[1];
            }
            partitions(mask | covered, candidates, cuts, sums, checked);
            for (sum, coefficient) in sums.iter_mut().zip(coefficients) {
                sum[0] -= coefficient[0];
                sum[1] -= coefficient[1];
            }
        }
    }
    let mut checked = 0;
    partitions(
        0,
        &candidates,
        &registry.cuts,
        &mut vec![[0, 0]; registry.cuts.len()],
        &mut checked,
    );
    assert!(checked > 0);
}

#[test]
fn explicit_arc_incidence_requires_exact_unit_master_rows() {
    use crate::master::model::{solve_with_master_variables, MasterVariable};
    use crate::master::preprocessing::Bounds;
    let (problem, config) = mixed_arc_problem_and_config();
    let mut options = Options::default();
    options.preprocessing.enabled = false;
    options.cg.time_limit = std::time::Duration::ZERO;
    let mut mismatched = config.clone();
    mismatched.graphs[0].arcs[0].packing_sets.clear();
    options.separation.rank_one_incidence = Some(mismatched);
    let error = super::super::solve(problem.clone(), options.clone())
        .err()
        .expect("incomplete incidence must be rejected");
    assert!(
        error.to_string().contains("differs from master rows"),
        "{error}"
    );

    options.separation.rank_one_incidence = Some(config.clone());
    let mut fractional = problem.clone();
    fractional.networks[0].arc_entries[0][0].1 = 0.5;
    let error = super::super::solve(fractional, options.clone())
        .err()
        .expect("fractional set incidence must be rejected");
    assert!(
        error.to_string().contains("master row arc coefficients"),
        "{error}"
    );

    let variable = MasterVariable {
        id: 100,
        cost: 0.,
        bounds: Bounds {
            lower: 0.,
            upper: 1.,
        },
        integer: false,
        entries: vec![(0, 1.)],
    };
    let error = solve_with_master_variables(problem, vec![variable], options)
        .err()
        .expect("independent contributions must be rejected");
    assert!(
        error.to_string().contains("exclusive unit equality rows"),
        "{error}"
    );
}
