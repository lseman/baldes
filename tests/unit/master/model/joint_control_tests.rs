use super::*;
use std::time::Duration;

#[test]
fn cyclic_suffix_bounds_reuse_stable_rows_for_large_hop_caps() {
    let mut graph = crate::graph::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut work = 64;
    let suffix = suffix_table(
        &graph,
        &[0., 1., 0.],
        1_000_000,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    assert!(suffix.stable);
    assert!(suffix.rows.len() < 10);
    assert_eq!(suffix.len(), 1_000_001);
    assert_eq!(suffix.get(1_000_000), suffix.rows.last());
    assert!(suffix.get(1_000_001).is_none());
    assert!(suffix.get(1_000_000).unwrap()[graph.source].is_some());
}

#[test]
fn cyclic_suffix_bounds_keep_distinct_rows_for_improving_cycles() {
    let mut graph = crate::graph::Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut work = 128;
    let suffix = suffix_table(&graph, &[0., -1., 0.], 8, &mut work, deadline).unwrap();
    assert!(!suffix.stable);
    assert_eq!(suffix.rows.len(), 9);
    assert_eq!(suffix.get(2).unwrap()[graph.source], Some(0.));
    assert_eq!(suffix.get(8).unwrap()[graph.source], Some(-6.));
    assert!(suffix.get(9).is_none());
}

#[test]
fn reduced_cost_interval_touching_zero_enumerates_paired_endpoints() {
    assert!(sign_ambiguous(Interval { lo: -1., hi: 0. }));
    assert!(sign_ambiguous(Interval { lo: -1., hi: 1. }));
    assert!(!sign_ambiguous(Interval {
        lo: -1.,
        hi: -f64::EPSILON
    }));
    assert!(!sign_ambiguous(Interval { lo: 0., hi: 1. }));
    assert!(!sign_ambiguous(Interval::point(0.)));
}

#[test]
fn bounded_walk_resource_pruning_uses_label_transition_semantics() {
    let mut graph = crate::graph::Graph::<2>::new(2, 1).unwrap();
    graph.initial[0] = Some(0.);
    graph.add_arc(0, 1, 0.).unwrap();
    graph.arcs[0].consumption[0] = -1.;
    let state = crate::graph::ResourceValues::from([0.]);

    assert_eq!(
        extend_bounded_walk_resources(&graph, 0, state),
        Some(Some(crate::graph::ResourceValues::from([0.])))
    );

    graph.resources[0] = crate::graph::ResourceKind::Equality;
    assert_eq!(extend_bounded_walk_resources(&graph, 0, state), Some(None));
}

#[cfg(feature = "multi-time-windows")]
#[test]
fn bounded_walk_resource_pruning_respects_window_gaps_and_arc_bounds() {
    let mut graph = crate::graph::Graph::<2>::new(2, 1).unwrap();
    graph.initial[0] = Some(0.);
    graph.add_arc(0, 1, 0.).unwrap();
    graph.arcs[0].consumption[0] = 4.;
    graph.arcs[0].bounds[0] = Some(crate::graph::Window::new(0., 5.).unwrap());
    graph
        .set_time_windows(
            1,
            0,
            &[
                crate::graph::Window::new(2., 3.).unwrap(),
                crate::graph::Window::new(7., 8.).unwrap(),
            ],
        )
        .unwrap();

    let state = crate::graph::ResourceValues::from([0.]);
    assert_eq!(extend_bounded_walk_resources(&graph, 0, state), Some(None));
}

fn paired_test_function(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
) -> crate::graph::coupled::PiecewiseLinearFunction {
    crate::graph::coupled::PiecewiseLinearFunction::new(vec![[x0, y0], [x1, y1]]).unwrap()
}

#[test]
fn unbounded_paired_terminal_bound_covers_negative_increments_and_work_limits() {
    let initial = paired_test_function(0., 10., 10., 10.);
    let arc = crate::graph::coupled::PairedResourceArc {
        arrival: paired_test_function(0., 0., 10., 10.),
        increment: paired_test_function(0., -2., 10., -2.),
        lower: paired_test_function(0., -100., 1e12, -100.),
        upper: paired_test_function(0., 1e12, 1e12, 1e12),
        waiting: 0,
    };
    let mut work = 10_000;
    let bound = paired_terminal_lower_bound(
        &initial,
        std::slice::from_ref(&arc),
        2,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    assert_eq!(bound, 6.);

    let mut no_work = 0;
    assert!(paired_terminal_lower_bound(
        &initial,
        std::slice::from_ref(&arc),
        2,
        &mut no_work,
        Instant::now() + Duration::from_secs(1),
    )
    .is_none());
}

#[test]
fn paired_terminal_lower_bound_stops_when_monotone_recurrence_rises() {
    let initial = paired_test_function(0., 10., 10., 10.);
    let arc = crate::graph::coupled::PairedResourceArc {
        arrival: paired_test_function(0., 0., 10., 10.),
        increment: paired_test_function(0., 2., 10., 2.),
        lower: paired_test_function(0., -100., 1e12, -100.),
        upper: paired_test_function(0., 1e12, 1e12, 1e12),
        waiting: 0,
    };
    let mut work = 100;
    let bound = paired_terminal_lower_bound(
        &initial,
        &[arc],
        1_000_000,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(bound, Some(10.));
    assert!(work > 0);
}

#[test]
fn paired_terminal_lower_bound_zero_hops_does_not_scan_arcs() {
    let initial = paired_test_function(0., 10., 10., 10.);
    let mut work = 8;
    let bound = paired_terminal_lower_bound(
        &initial,
        &[],
        0,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(bound, Some(10.));
}

#[test]
fn paired_negative_terminal_ceiling_applies_lower_resets_per_hop() {
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut work = 8;
    let ceiling =
        paired_terminal_ceiling_after_hops(0., 1., 0., 5., 100., false, 2, &mut work, deadline)
            .unwrap();
    assert_eq!(ceiling, 6.);
    assert_eq!(work, 6);

    let mut capped_work = 3;
    let capped = paired_terminal_ceiling_after_hops(
        8.,
        2.,
        0.,
        0.,
        9.,
        false,
        3,
        &mut capped_work,
        deadline,
    )
    .unwrap();
    assert_eq!(capped, 9.);

    let mut unbounded_work = 2;
    let unbounded = paired_terminal_ceiling_after_hops(
        0.,
        6e11,
        0.,
        0.,
        1e12,
        true,
        2,
        &mut unbounded_work,
        deadline,
    )
    .unwrap();
    assert_eq!(unbounded, 1.2e12);

    let mut exhausted = 1;
    assert!(paired_terminal_ceiling_after_hops(
        0.,
        1.,
        0.,
        5.,
        100.,
        false,
        2,
        &mut exhausted,
        deadline,
    )
    .is_none());
}

#[test]
fn paired_terminal_ceiling_stops_when_monotone_recurrence_falls() {
    let mut work = 2;
    let result = paired_terminal_ceiling_after_hops(
        10.,
        -1.,
        0.,
        -100.,
        100.,
        false,
        1_000_000,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(result, Some(10.));
    assert_eq!(work, 1);
}

#[test]
fn paired_terminal_upper_bound_stops_when_monotone_recurrence_falls() {
    let initial = paired_test_function(0., 10., 10., 10.);
    let arc = crate::graph::coupled::PairedResourceArc {
        arrival: paired_test_function(0., 0., 10., 10.),
        increment: paired_test_function(0., -1., 10., -1.),
        lower: paired_test_function(0., -100., 10., -100.),
        upper: paired_test_function(0., 100., 10., 100.),
        waiting: 0,
    };
    let mut work = 100;
    let bound = paired_terminal_upper_bound(
        &initial,
        &[arc],
        1_000_000,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    assert_eq!(bound.terminal, 10.);
    assert!(work > 0);
}

#[test]
fn unbounded_paired_terminal_upper_bound_ignores_shift_sentinel_as_a_cap() {
    let initial = paired_test_function(0., 0., 10., 0.);
    let arc = crate::graph::coupled::PairedResourceArc {
        arrival: paired_test_function(0., 0., 10., 10.),
        increment: paired_test_function(0., 6e11, 10., 6e11),
        lower: paired_test_function(0., 0., 10., 0.),
        upper: paired_test_function(0., 1e12, 1e12, 1e12),
        waiting: 0,
    };
    let mut work = 1_000;
    let bound = paired_terminal_upper_bound(
        &initial,
        std::slice::from_ref(&arc),
        2,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    )
    .unwrap();
    assert_eq!(bound.terminal, 1.2e12);
    assert!(bound.upper_unbounded);

    let mut no_work = 0;
    assert!(paired_terminal_upper_bound(
        &initial,
        std::slice::from_ref(&arc),
        2,
        &mut no_work,
        Instant::now() + Duration::from_secs(1),
    )
    .is_none());
}

#[test]
fn paired_base_domain_pruning_keeps_hulls_and_proves_empty_departure_domains() {
    use crate::graph::coupled::{
        CoupledResource, PairedCostNormalization, PairedResourceArc, PairedResourceMode,
    };
    let mut graph = crate::graph::Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            initial: paired_test_function(0., 0., 10., 0.),
            initial_base: Window::new(0., 10.).unwrap(),
            arcs: vec![PairedResourceArc {
                arrival: paired_test_function(0., 0., 10., 10.),
                increment: paired_test_function(0., 0., 10., 0.),
                lower: paired_test_function(0., 0., 10., 0.),
                upper: paired_test_function(0., 100., 10., 100.),
                waiting: 0,
            }],
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
        }])
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(1);
    let mut work = 100;
    let mut hulls = HashMap::new();
    let roots = paired_prefix_roots(&graph, &mut work, deadline).unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].resource, 0);
    assert_eq!(roots[0].base.lo, 0.);
    assert_eq!(roots[0].base.hi, 10.);
    assert_eq!(roots[0].dependent_lower, 0.);
    let narrowed = [PairedPrefixDomain {
        resource: 0,
        base: Interval { lo: 2., hi: 4. },
        dependent_lower: 0.,
        dependent_upper: 0.,
    }];
    let reachable =
        extend_paired_prefix_domains(&graph, 0, &narrowed, &mut hulls, &mut work, deadline)
            .unwrap()
            .unwrap();
    assert_eq!(reachable.len(), 1);
    assert_eq!(reachable[0].base.lo, 2.);
    assert_eq!(reachable[0].base.hi, 10.);
    assert_eq!(reachable[0].dependent_lower, 0.);
    assert_eq!(reachable[0].dependent_upper, 0.);

    let impossible = [PairedPrefixDomain {
        resource: 0,
        base: Interval { lo: 20., hi: 30. },
        dependent_lower: 0.,
        dependent_upper: 0.,
    }];
    assert!(matches!(
        extend_paired_prefix_domains(&graph, 0, &impossible, &mut hulls, &mut work, deadline,),
        Some(None)
    ));
}

#[test]
fn front_algebra_paired_prefix_keeps_endpoint_snapped_head_below_arrival() {
    use crate::graph::coupled::{
        CoupledResource, FrontOperationPolicy, PairedCostNormalization, PairedResourceMode,
        PreparedSlopeFunction, SlopeRelationArc,
    };

    let function =
        |value| PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (0., 1.)]).unwrap();
    let mut graph = crate::graph::Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(0.),
            initial_base: Window::new(0., 1.).unwrap(),
            arcs: vec![SlopeRelationArc {
                arrival: PreparedSlopeFunction::from_prepared_pairs(vec![(0.00005, 0.), (1., 1.)])
                    .unwrap(),
                increment: function(0.),
                lower: function(0.),
                upper: function(10.),
                waiting: 1,
            }],
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::FrontAlgebra,
        }])
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(1);
    let mut work = 1_000;
    let mut hulls = HashMap::new();
    let roots = paired_prefix_roots(&graph, &mut work, deadline).unwrap();
    assert_eq!(roots.len(), 1);

    let next = extend_paired_prefix_domains(&graph, 0, &roots, &mut hulls, &mut work, deadline)
        .unwrap()
        .unwrap();
    assert_eq!(next.len(), 1);
    assert!(next[0].base.lo < 0.);
    assert!(next[0].dependent_lower <= 0.);
    assert_eq!(hulls.len(), 1);
    let cached = extend_paired_prefix_domains(&graph, 0, &roots, &mut hulls, &mut work, deadline)
        .unwrap()
        .unwrap();
    assert_eq!(cached[0].base.lo.to_bits(), next[0].base.lo.to_bits());
    assert_eq!(cached[0].base.hi.to_bits(), next[0].base.hi.to_bits());
    assert_eq!(
        cached[0].dependent_lower.to_bits(),
        next[0].dependent_lower.to_bits()
    );
    assert_eq!(
        cached[0].dependent_upper.to_bits(),
        next[0].dependent_upper.to_bits()
    );
    let mut exhausted = 0;
    assert!(
        extend_paired_prefix_domains(&graph, 0, &roots, &mut hulls, &mut exhausted, deadline,)
            .is_none()
    );
    let mut work = 1_000;
    assert!(extend_paired_prefix_domains(
        &graph,
        0,
        &roots,
        &mut hulls,
        &mut work,
        Instant::now() - Duration::from_millis(1),
    )
    .is_none());
}

fn root() -> Output<2> {
    let mut source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/joint-interior-model.json"
    ))
    .unwrap();
    source["networks"][0]["resources"] = serde_json::json!([
        {"kind":"disposable","bounds":[[0.,0.],[0.,10.]]}
    ]);
    let (problem, registry) = format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.integer_primal_termination = true;
    options.cg.stabilize = false;
    registry.configure_pricing(&mut options);
    solve(problem, options).unwrap()
}

#[test]
fn joint_proposal_witness_requires_exact_whole_model_rows_and_residual_caps() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let tag = |values| {
        state
            .routes
            .iter()
            .position(|route| route.route.coupled_values == values)
            .unwrap()
    };
    let tags = [
        tag(vec![(0, 0.), (1, 0.)]) as i32,
        tag(vec![(0, 8.), (1, 8.)]) as i32,
    ];
    let interior = tag(vec![(0, 4.), (1, 4.)]);
    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 1., &[]).unwrap();
    for _ in 0..2 {
        let column = master.add_column(999., &[(0, 1.)]).unwrap();
        master.set_upper(column, 0.5).unwrap();
    }
    assert!(master.solve());
    let metadata = |column| tags.get(column).copied();
    let witness = repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .unwrap();
    assert_eq!(witness.0, -10.);
    assert_eq!(
        witness.1[0].route.route.coupled_values,
        vec![(0, 4.), (1, 4.)]
    );
    assert!(
        !output
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap()
            .enumeration_report()
            .complete
    );
    state.route_upper_bounds.insert(interior, 0.);
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
    state.route_upper_bounds.clear();
    state.shared_rows[0].rhs = 9.;
    let changed = repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .unwrap();
    assert_eq!(
        changed.1[0].route.route.coupled_values,
        vec![(0, 4.5), (1, 4.5)]
    );
    state.shared_rows[0].rhs = 17.;
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
    state.shared_rows[0].rhs = 8.;
    state.universe = Some(vec![tags[0] as usize, tags[1] as usize]);
    assert!(repair_witness(&state, &master, metadata, 1e-7)
        .unwrap()
        .is_none());
}

#[test]
fn joint_control_targets_share_work_iterations_and_deadlines() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let oracle = &state.oracles[0];
    let mut work = 1_000_000;
    let mut iterations = 100_000;
    let priced = path(
        oracle,
        state.shared_offset,
        &vec![0.; state.rhs.len()],
        &[0],
        &mut work,
        &mut iterations,
        state.deadline,
    )
    .unwrap()
    .unwrap();
    let initial = priced
        .route
        .as_ref()
        .unwrap()
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(priced.greedy.clone()))
        .unwrap();
    assert!(joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut 0, &mut iterations),
        state.deadline,
    )
    .is_none());
    assert!(joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut 0),
        state.deadline,
    )
    .is_none());
    let proposals = joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
        state.deadline,
    )
    .unwrap();
    let route = initial
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(proposals[0].clone()))
        .unwrap();
    assert_eq!(route.coupled_values, vec![(0, 4.), (1, 4.)]);
    state.deadline = Instant::now();
    assert!(joint_controls(
        &state,
        &state.oracles[0],
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
        state.deadline,
    )
    .is_none());
}

#[test]
fn joint_control_proposals_include_attainable_extremes_of_an_equality_face() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let oracle = &state.oracles[0];
    let mut work = 1_000_000;
    let mut iterations = 100_000;
    let priced = path(
        oracle,
        state.shared_offset,
        &vec![0.; state.rhs.len()],
        &[0],
        &mut work,
        &mut iterations,
        state.deadline,
    )
    .unwrap()
    .unwrap();
    let initial = priced
        .route
        .as_ref()
        .unwrap()
        .clone()
        .with_paired_controls(&oracle.network.graph, sorted(priced.greedy.clone()))
        .unwrap();

    // Gamma0 + Gamma1 = 8 leaves a line segment. The other row is made
    // redundant so optimizing either coordinate exposes both face endpoints.
    state.shared_rows[1].sense = RowSense::LessEqual;
    state.shared_rows[1].rhs = 8.;
    state.rhs[state.shared_offset + 1] = 8.;
    let proposals = joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
        state.deadline,
    )
    .expect("bounded LP face should produce source-replayed proposals");
    let values: Vec<_> = proposals
        .into_iter()
        .map(|controls| {
            initial
                .clone()
                .with_paired_controls(&oracle.network.graph, sorted(controls))
                .unwrap()
                .coupled_values
        })
        .collect();

    assert!(values.contains(&vec![(0, 0.), (1, 8.)]));
    assert!(values.contains(&vec![(0, 8.), (1, 0.)]));
}

#[test]
fn global_joint_target_objectives_cover_coordinate_extremes() {
    let domains = [Interval { lo: 0., hi: 8. }; 2];
    let rows = [(RowSense::Equal, 8., vec![(0, 1.), (1, 1.)])];
    let costs = [0., 0.];
    let proposal_count = costs.len() * 2 + 2;
    assert_eq!(proposal_count, 6);

    let mut iterations = 100_000;
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let targets: Vec<_> = (0..proposal_count)
        .filter_map(|proposal| {
            let objective = terminal_objective(&costs, proposal)?;
            solve_terminal_targets(&domains, &rows, &objective, &mut iterations, deadline)
        })
        .collect();

    assert!(targets.contains(&vec![0., 8.]));
    assert!(targets.contains(&vec![8., 0.]));
}

#[test]
fn terminal_vertex_sweep_finds_joint_extrema_missed_by_coordinate_objectives() {
    let domains = [0.; 3].map(|_| Interval { lo: 0., hi: 1. });
    let rows = vec![
        (RowSense::LessEqual, 1., vec![(0, 1.), (1, 1.)]),
        (RowSense::LessEqual, 1., vec![(1, 1.), (2, 1.)]),
        (RowSense::LessEqual, 1., vec![(0, 1.), (2, 1.)]),
    ];
    let mut work = 100_000;
    let vertices = terminal_vertices(
        &domains,
        &rows,
        &mut work,
        Instant::now() + Duration::from_secs(1),
    );
    assert!(vertices
        .iter()
        .any(|point| { point.iter().all(|value| (*value - 0.5).abs() <= 1e-9) }));
}

#[test]
fn terminal_vertex_sweep_uses_shared_work_instead_of_a_fixed_vertex_cap() {
    let count = 140;
    let domains = [Interval { lo: 0., hi: 1. }; 2];
    let rows: Vec<_> = (0..count)
        .map(|index| {
            let angle = 2. * std::f64::consts::PI * index as f64 / count as f64;
            let a = angle.cos();
            let b = angle.sin();
            let rhs = 0.5 * (a + b) + 0.4 * (std::f64::consts::PI / count as f64).cos();
            (RowSense::LessEqual, rhs, vec![(0, a), (1, b)])
        })
        .collect();
    let mut work = 100_000;
    let vertices = terminal_vertices(
        &domains,
        &rows,
        &mut work,
        Instant::now() + Duration::from_secs(2),
    );
    assert!(
        vertices.len() >= 129,
        "only found {} vertices",
        vertices.len()
    );
    assert!(vertices
        .iter()
        .flatten()
        .all(|value| (0. ..=1.).contains(value)));
}

#[test]
fn joint_control_recovers_a_three_resource_vertex_missed_by_coordinate_targets() {
    let mut source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/coupled/joint-interior-model.json"
    ))
    .unwrap();
    source["rows"] = serde_json::json!([
        {"sense":"le", "rhs":10.},
        {"sense":"le", "rhs":10.},
        {"sense":"le", "rhs":10.}
    ]);
    source["networks"][0]["resources"] = serde_json::json!([
        {"kind":"disposable", "bounds":[[0.,0.],[0.,10.]]}
    ]);
    let mut definitions = source["networks"][0]["coupled_resources"]
        .as_array()
        .unwrap()
        .clone();
    definitions.push(definitions[0].clone());
    source["networks"][0]["coupled_resources"] = serde_json::json!(definitions);
    source["pricing_variables"] = serde_json::json!([
        {"network":42, "id":901, "cost":0., "accumulated_resource":0,
         "entries":[[0,1.],[2,1.]]},
        {"network":42, "id":902, "cost":0., "accumulated_resource":1,
         "entries":[[0,1.],[1,1.]]},
        {"network":42, "id":903, "cost":0., "accumulated_resource":2,
         "entries":[[1,1.],[2,1.]]}
    ]);
    let (problem, registry) = format::Model::from_json(&source.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = Options::default();
    options.integer_primal_termination = true;
    options.cg.stabilize = false;
    registry.configure_pricing(&mut options);
    let output = solve(problem, options).unwrap();
    let state = output.snapshot.state.fork_state();
    let oracle = &state.oracles[0];
    let mut work = 1_000_000;
    let mut iterations = 100_000;
    let priced = path(
        oracle,
        state.shared_offset,
        &vec![0.; state.rhs.len()],
        &[0],
        &mut work,
        &mut iterations,
        state.deadline,
    )
    .unwrap()
    .unwrap();
    assert_eq!(priced.resources, vec![0, 1, 2]);
    let graph = oracle.network.graph.clone();
    let initial = priced
        .route
        .as_ref()
        .unwrap()
        .clone()
        .with_paired_controls(graph.as_ref(), sorted(priced.greedy.clone()))
        .unwrap();
    let proposals = joint_controls(
        &state,
        oracle,
        &[0],
        &priced,
        &initial,
        1.,
        (&mut work, &mut iterations),
        state.deadline,
    )
    .expect("the jointly feasible terminal vertex should be proposed");
    let values: Vec<_> = proposals
        .into_iter()
        .map(|controls| {
            initial
                .clone()
                .with_paired_controls(graph.as_ref(), sorted(controls))
                .unwrap()
                .coupled_values
        })
        .collect();
    assert!(values.contains(&vec![(0, 5.), (1, 5.), (2, 5.)]));
}

#[test]
fn joint_control_projects_one_resource_through_opposing_rows() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let mut work = 1_000_000;
    let mut iterations = 100_000;
    let priced = {
        let oracle = &state.oracles[0];
        path(
            oracle,
            state.shared_offset,
            &vec![0.; state.rhs.len()],
            &[0],
            &mut work,
            &mut iterations,
            state.deadline,
        )
        .unwrap()
        .unwrap()
    };
    let graph = state.oracles[0].network.graph.clone();
    let initial = priced
        .route
        .as_ref()
        .unwrap()
        .clone()
        .with_paired_controls(graph.as_ref(), sorted(priced.greedy.clone()))
        .unwrap();

    let gamma_row = state.shared_offset;
    let upper_row = state.shared_offset + 1;
    state.shared_rows[0].sense = RowSense::GreaterEqual;
    state.shared_rows[0].rhs = 4.;
    state.shared_rows[1].sense = RowSense::LessEqual;
    state.shared_rows[1].rhs = 6.;
    state.rhs[gamma_row] = 4.;
    state.rhs[upper_row] = 6.;
    state.oracles[0]
        .activity_rows
        .retain(|entry| entry.resource == 0);

    let mut one_resource = priced;
    one_resource.resources.truncate(1);
    one_resource.domains.truncate(1);
    one_resource.minimum.truncate(1);
    one_resource.maximum.truncate(1);
    one_resource.greedy.truncate(1);
    let oracle = &state.oracles[0];
    let proposals = joint_controls(
        &state,
        oracle,
        &[0],
        &one_resource,
        &initial,
        1.,
        (&mut work, &mut iterations),
        state.deadline,
    )
    .expect("the common interval between opposing rows has attained endpoints");
    let values: Vec<_> = proposals
        .into_iter()
        .filter_map(|controls| {
            initial
                .clone()
                .with_paired_controls(graph.as_ref(), sorted(controls))
                .ok()?
                .coupled_values
                .into_iter()
                .find(|(resource, _)| *resource == 0)
                .map(|(_, value)| value)
        })
        .collect();

    assert!(values.contains(&4.));
    assert!(values.contains(&6.));
}

#[test]
fn scheduled_repair_checks_caps_and_coverage_after_profile_changes() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let original = state
        .routes
        .iter()
        .position(|r| r.route.coupled_values == vec![(0, 4.), (1, 4.)])
        .unwrap();
    let mut changed = state.routes[original].as_ref().clone();
    changed.route.resource_profile.last_mut().unwrap()[0] = 5.;
    let copy = FixedRoute {
        route: changed.clone(),
        quantity: 1.,
    };
    let tag = state.routes.len();
    state.routes.push(Arc::new(changed));
    state.route_upper_bounds.insert(tag, 0.);
    assert!(!scheduled_identities_allowed(&state, &[copy.clone()]));
    // Fixed copies are outside this residual-cap check, including a consumed cap0.
    assert!(scheduled_identities_allowed(&state, &[]));
    state.route_upper_bounds.insert(tag, 1.);
    assert!(scheduled_identities_allowed(&state, &[copy.clone()]));
    assert!(!scheduled_identities_allowed(
        &state,
        &[copy.clone(), copy.clone()]
    ));
    state.universe = Some(vec![original]);
    assert!(!scheduled_identities_allowed(&state, &[copy.clone()]));
    state.universe.as_mut().unwrap().push(tag);
    assert!(scheduled_identities_allowed(&state, &[copy.clone()]));
    state.route_upper_bounds.insert(tag, f64::INFINITY);
    assert!(scheduled_identities_allowed(&state, &[copy]));
}

#[test]
fn paired_witness_allocations_split_counts_across_capped_identities() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        .unwrap();
    assert_eq!(
        state.routes[first].route.arcs,
        state.routes[second].route.arcs
    );
    state.route_upper_bounds.insert(first, 1.);
    state.route_upper_bounds.insert(second, 1.);
    let choices = [first, second].map(|tag| (tag, 0.5, state.routes[tag].as_ref().clone()));

    let allocations = identity_allocations(&state, &choices, 2., 128, state.deadline);

    assert!(allocations.iter().any(|allocation| {
        allocation.len() == 2
            && allocation.iter().all(|route| route.quantity == 1.)
            && allocation
                .iter()
                .any(|route| route.route.route.coupled_values == vec![(0, 0.), (1, 0.)])
            && allocation
                .iter()
                .any(|route| route.route.route.coupled_values == vec![(0, 8.), (1, 8.)])
    }));
}

#[test]
fn joint_witness_seed_search_reaches_later_identity_allocations() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        .unwrap();
    assert_eq!(
        state.routes[first].route.arcs,
        state.routes[second].route.arcs
    );
    state.route_upper_bounds.insert(first, 1.);
    state.route_upper_bounds.insert(second, 1.);
    let choices = vec![first, second]
        .into_iter()
        .map(|tag| {
            (
                tag,
                if tag == first { 1. } else { 0. },
                state.routes[tag].as_ref().clone(),
            )
        })
        .collect::<Vec<_>>();
    let groups = [(choices, 1.)];
    let mut work = 100;
    let mut iterations = 100;
    let mut selected = Vec::new();
    let mut seen = Vec::new();

    let found = for_each_joint_witness_seed(
        &state,
        &groups,
        0,
        &mut selected,
        &mut work,
        &mut iterations,
        &mut |seed, _, _| {
            seen.push(seed[0].route.route.coupled_values.clone());
            Ok(seed[0].route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        },
    )
    .unwrap();

    assert!(found);
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1], vec![(0, 8.), (1, 8.)]);
}

#[test]
fn lazy_identity_search_reaches_allocations_beyond_the_old_shortlist() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let template = state.routes[0].as_ref().clone();
    let mut choices = Vec::new();
    for index in 0..130 {
        let tag = state.routes.len();
        let mut route = template.clone();
        route.route.coupled_values = vec![(0, index as f64)];
        state.routes.push(Arc::new(route.clone()));
        state.route_upper_bounds.insert(tag, 1.);
        choices.push((tag, 0.5, route));
    }

    let mut saw_first_registered_identity = false;
    let mut work = usize::MAX;
    let stopped = for_each_identity_allocation(
        &state,
        &choices,
        1.,
        state.deadline,
        &mut work,
        &mut |allocation, _deadline, _work| {
            saw_first_registered_identity |= allocation
                .iter()
                .any(|route| route.route.route.coupled_values == vec![(0, 0.)]);
            Ok(saw_first_registered_identity)
        },
    )
    .unwrap();

    assert!(stopped);
    assert!(saw_first_registered_identity);
}

#[test]
fn lazy_identity_search_streams_past_one_hundred_thousand_allocations() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    let template = state.routes[0].as_ref().clone();
    let mut choices = Vec::new();
    for _ in 0..20 {
        let tag = state.routes.len();
        state.routes.push(Arc::new(template.clone()));
        state.route_upper_bounds.insert(tag, 1.);
        choices.push((tag, 0.5, template.clone()));
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut allocations_seen = 0;
    let mut work = usize::MAX;
    let stopped = for_each_identity_allocation(
        &state,
        &choices,
        10.,
        deadline,
        &mut work,
        &mut |_allocation, _deadline, _work| {
            allocations_seen += 1;
            Ok(allocations_seen == 100_001)
        },
    )
    .unwrap();

    assert!(stopped);
    assert_eq!(allocations_seen, 100_001);
}

#[test]
fn identity_allocation_visit_stops_at_its_shared_work_budget() {
    let output = root();
    let state = output.snapshot.state.fork_state();
    let tag = 0;
    let choices = [(tag, 0.5, state.routes[tag].as_ref().clone())];
    let mut work = 2;
    let mut allocations_seen = 0;

    let stopped = for_each_identity_allocation(
        &state,
        &choices,
        1.,
        state.deadline,
        &mut work,
        &mut |_allocation, _deadline, _work| {
            allocations_seen += 1;
            Ok(false)
        },
    )
    .unwrap();

    assert!(stopped);
    assert_eq!(work, 0);
    assert!(allocations_seen <= 2);
}

#[test]
fn paired_witness_allocations_search_nonendpoint_integer_splits() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    state.oracles[0].network.multiplicity = Window::new(10., 10.).unwrap();
    state.oracles[0].total_multiplicity = Window::new(10., 10.).unwrap();
    state.oracles[0].proved_multiplicity_upper = None;
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        .unwrap();
    state.route_upper_bounds.insert(first, 8.);
    state.route_upper_bounds.insert(second, 8.);
    assert_eq!(enumeration::route_upper(&state, first), Some(8.));
    assert_eq!(enumeration::route_upper(&state, second), Some(8.));
    let choices = [first, second].map(|tag| (tag, 0.5, state.routes[tag].as_ref().clone()));

    let allocations = identity_allocations(&state, &choices, 10., 128, state.deadline);

    assert!(allocations.iter().any(|allocation| {
        allocation.len() == 2
            && allocation.iter().any(|route| {
                route.quantity == 4. && route.route.route.coupled_values == vec![(0, 0.), (1, 0.)]
            })
            && allocation.iter().any(|route| {
                route.quantity == 6. && route.route.route.coupled_values == vec![(0, 8.), (1, 8.)]
            })
    }));
}

#[test]
fn paired_witness_repair_recovers_path_count_from_capped_identities() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    state.oracles[0].total_multiplicity = Window::new(2., 2.).unwrap();
    let first = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 0.), (1, 0.)])
        .unwrap();
    let second = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 4.), (1, 4.)])
        .unwrap();
    let arcs = state.routes[first].route.arcs.clone();
    assert_eq!(arcs, state.routes[second].route.arcs);
    for (tag, route) in state.routes.iter().enumerate() {
        if route.network == state.routes[first].network && route.route.arcs == arcs {
            state.route_upper_bounds.insert(tag, 1.);
        }
    }

    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    let tags = [first, second, first, second].map(|tag| tag as i32);
    for _ in 0..4 {
        let column = master.add_column(999., &[(0, 1.)]).unwrap();
        master.set_upper(column, 0.5).unwrap();
    }
    assert!(master.solve());

    let witness = repair_witness(&state, &master, |column| tags.get(column).copied(), 1e-7)
        .unwrap()
        .expect("the path count of two can use two identities capped at one each");

    assert_eq!(
        witness.1.iter().map(|route| route.quantity).sum::<f64>(),
        2.
    );
    assert!(witness
        .1
        .iter()
        .any(|route| route.route.route.coupled_values == vec![(0, 0.), (1, 0.)]));
    assert!(witness
        .1
        .iter()
        .any(|route| route.route.route.coupled_values == vec![(0, 4.), (1, 4.)]));
}

#[test]
fn paired_witness_repair_can_split_with_the_minimum_front_identity() {
    let output = root();
    let mut state = output.snapshot.state.fork_state();
    state.oracles[0].total_multiplicity = Window::new(2., 2.).unwrap();
    let maximum = state
        .routes
        .iter()
        .position(|route| route.route.coupled_values == vec![(0, 8.), (1, 8.)])
        .unwrap();
    let arcs = state.routes[maximum].route.arcs.clone();
    let graph = state.oracles[0].network.graph.as_ref().clone();
    let problem = crate::master::routing::RoutingContext::set_partitioning(graph, 1.).unwrap();
    let default_route = problem.elementary_route(&arcs).unwrap();
    assert_eq!(default_route.coupled_values, vec![(0, 0.), (1, 0.)]);
    let minimum = state.routes.len();
    state.routes.push(Arc::new(NetworkRoute {
        network: state.routes[maximum].network,
        route: default_route,
    }));
    assert_eq!(arcs, state.routes[maximum].route.arcs);
    for (tag, route) in state.routes.iter().enumerate() {
        if route.network == state.routes[minimum].network && route.route.arcs == arcs {
            state.route_upper_bounds.insert(tag, 1.);
        }
    }

    let mut master = ClpMaster::new().unwrap();
    master.add_row(RowSense::Equal, 2., &[]).unwrap();
    let tags = [minimum, maximum, minimum, maximum].map(|tag| tag as i32);
    for _ in 0..4 {
        let column = master.add_column(999., &[(0, 1.)]).unwrap();
        master.set_upper(column, 0.5).unwrap();
    }
    assert!(master.solve());

    let witness = repair_witness(&state, &master, |column| tags.get(column).copied(), 1e-7)
        .unwrap()
        .expect("the minimum-front route is a valid paired identity candidate");

    assert_eq!(
        witness.1.iter().map(|route| route.quantity).sum::<f64>(),
        2.
    );
    assert!(witness
        .1
        .iter()
        .any(|route| route.route.route.coupled_controls.is_empty()));
    assert!(witness
        .1
        .iter()
        .any(|route| !route.route.route.coupled_controls.is_empty()));
}

#[test]
fn joint_recovery_keeps_fixed_copies_and_signed_master_contributions() {
    use crate::graph::coupled::PairedControl;
    let output = root();
    for sign in [1_f64, -1.] {
        let mut state = output.snapshot.state.fork_state();
        let tag = |values| {
            state
                .routes
                .iter()
                .position(|r| r.route.coupled_values == values)
                .unwrap() as i32
        };
        let tags = [tag(vec![(0, 0.), (1, 0.)]), tag(vec![(0, 8.), (1, 8.)])];
        let mut fixed = state.routes[0].as_ref().clone();
        fixed.route = fixed
            .route
            .with_paired_controls(
                &state.oracles[0].network.graph,
                vec![
                    PairedControl {
                        resource: 0,
                        base: vec![2., 3.],
                    },
                    PairedControl {
                        resource: 1,
                        base: vec![2., 3.],
                    },
                ],
            )
            .unwrap();
        state.fixed_routes.push(FixedRoute {
            route: fixed,
            quantity: 1.,
        });
        state.oracles[0].total_multiplicity = Window::new(2., 2.).unwrap();
        state.shared_rows[0].rhs = 13. + sign;
        state
            .master_variables
            .push(master_variables::MasterVariable {
                id: 77,
                cost: 2.,
                bounds: crate::master::preprocessing::Bounds {
                    lower: sign.min(0.),
                    upper: sign.max(0.),
                },
                integer: false,
                entries: vec![(0, 2.)],
            });
        let mut master = ClpMaster::new().unwrap();
        master.add_row(RowSense::Equal, 1., &[]).unwrap();
        master.add_row(RowSense::Equal, 0.5, &[]).unwrap();
        for _ in 0..2 {
            let column = master.add_column(999., &[(0, 1.)]).unwrap();
            master.set_upper(column, 0.5).unwrap();
        }
        master.add_column(999., &[(1, 1.)]).unwrap();
        master.add_column(0., &[]).unwrap();
        assert!(master.solve());
        let metadata = |column| match column {
            0 | 1 => Some(tags[column]),
            2 => Some(if sign > 0. { -2 } else { -3 }),
            _ => None,
        };
        // Numerical artificial residuals do not become original source activity.
        master.values[3] = -1e-12;
        let witness = repair_witness(&state, &master, metadata, 1e-7)
            .unwrap()
            .unwrap();
        assert_eq!(witness.0, -20. + sign);
        assert_eq!(witness.2, vec![(77, sign * 0.5)]);
        assert_eq!(witness.1.len(), 2);
        assert_eq!(
            witness.1[0].route.route.coupled_values,
            vec![(0, 2.), (1, 2.)]
        );
        assert_eq!(
            witness.1[1].route.route.coupled_values,
            vec![(0, 4.5), (1, 4.5)]
        );
        for artifact in [-0.5, 0.5] {
            master.values[3] = artifact;
            assert!(repair_witness(&state, &master, metadata, 1e-7)
                .unwrap()
                .is_none());
        }
    }
}
