use super::*;

fn model() -> Solver<2> {
    let mut solver = Solver::new(4, 1).unwrap();
    for v in 0..4 {
        solver.graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    let a = solver.graph.add_arc(0, 1, 0.).unwrap();
    solver.graph.arc_mut(a).unwrap().variables = vec![(0, 1e8)];
    solver.graph.add_arc(1, 3, 0.).unwrap();
    solver.graph.add_arc(0, 2, 0.05).unwrap();
    solver.graph.add_arc(2, 3, 0.).unwrap();
    solver.maximum_variable_id = Some(0);
    solver
        .set_option("BALDESuseBidirectionalSearch", 0.)
        .unwrap();
    solver
}

fn fresh(solver: &Solver<2>) -> PricingResult {
    let mut raw = solver.fork_pricing(false);
    for arc in &mut raw.graph.arcs {
        arc.reduced_cost = arc.variables.iter().fold(arc.cost, |sum, &(id, a)| {
            sum + solver.variable_costs[id] * a
        });
    }
    raw.price_phase(0, 8, false).unwrap()
}

fn matches_fresh(solver: &mut Solver<2>, cost: f64) {
    solver.cache_variable_costs(&[cost]).unwrap();
    let expected = fresh(solver);
    let cached: Vec<_> = solver.graph.arcs.iter().map(|a| a.reduced_cost).collect();
    let actual = solver.price_functor_phase(0, 8).unwrap();
    assert!(actual.exact);
    assert_eq!(
        actual.minimum_reduced_cost.to_bits(),
        expected.minimum_reduced_cost.to_bits()
    );
    assert_eq!(
        actual
            .columns
            .iter()
            .map(|r| (&r.arcs, r.cost.to_bits()))
            .collect::<Vec<_>>(),
        expected
            .columns
            .iter()
            .map(|r| (&r.arcs, r.cost.to_bits()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        cached,
        solver
            .graph
            .arcs
            .iter()
            .map(|a| a.reduced_cost)
            .collect::<Vec<_>>()
    );
    let raw = &solver.raw_pricing.as_ref().unwrap().solver;
    assert!(raw.variable_costs.is_empty());
    assert!(raw.enumeration.is_none());
    let engine = raw.engine.as_ref().unwrap();
    assert_eq!(engine.forward.labels.len(), 0);
    assert_eq!(engine.backward.labels.len(), 0);
    assert_eq!(engine.forward.labels.allocated_bytes(), 0);
    assert_eq!(engine.backward.labels.allocated_bytes(), 0);
}

#[test]
fn raw_workspace_reuses_preparation_but_not_labels_or_cost_dependent_pruning() {
    let mut solver = model();
    matches_fresh(&mut solver, 1e-9);
    let prepared = solver
        .raw_pricing
        .as_ref()
        .unwrap()
        .solver
        .prepared
        .clone()
        .unwrap();
    for cost in [0., -1e-9, 1e-8, 1e-9] {
        matches_fresh(&mut solver, cost);
        assert!(Arc::ptr_eq(
            &prepared,
            solver
                .raw_pricing
                .as_ref()
                .unwrap()
                .solver
                .prepared
                .as_ref()
                .unwrap()
        ));
    }
    let child = solver.fork_for_node();
    assert!(child.raw_pricing.is_none());
    assert_eq!(child.variable_costs, solver.variable_costs);
}

#[test]
fn raw_workspace_invalidates_on_topology_restrictions_presence_and_configuration() {
    use crate::cuts::soft_conflicts::{Mode, PresenceRow, PricingInput};
    let mut solver = model();
    matches_fresh(&mut solver, 1e-9);
    let first = solver
        .raw_pricing
        .as_ref()
        .unwrap()
        .solver
        .prepared
        .clone()
        .unwrap();
    solver.graph.arcs[2].cost = 0.03;
    matches_fresh(&mut solver, 1e-9);
    assert!(!Arc::ptr_eq(
        &first,
        solver
            .raw_pricing
            .as_ref()
            .unwrap()
            .solver
            .prepared
            .as_ref()
            .unwrap()
    ));
    solver.set_forbidden_arcs(vec![2]).unwrap();
    matches_fresh(&mut solver, 1e-9);
    solver.set_forbidden_arcs(vec![]).unwrap();
    matches_fresh(&mut solver, 1e-9);
    solver
        .set_presence_rows(vec![PricingInput {
            row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
            arc_events: vec![vec![0], vec![1], vec![], vec![]],
            dual: 0.2,
        }])
        .unwrap();
    matches_fresh(&mut solver, 1e-9);
    let before = solver
        .raw_pricing
        .as_ref()
        .unwrap()
        .solver
        .prepared
        .clone()
        .unwrap();
    solver
        .set_option("BALDESnumberOfBucketsPerVertex", 31.)
        .unwrap();
    matches_fresh(&mut solver, 1e-9);
    assert!(!Arc::ptr_eq(
        &before,
        solver
            .raw_pricing
            .as_ref()
            .unwrap()
            .solver
            .prepared
            .as_ref()
            .unwrap()
    ));
}

#[test]
fn raw_workspace_refreshes_cut_shapes_duals_and_rollback_budgets() {
    use crate::cuts::{CutClass, MemoryKind, RankOneCut};
    let mut solver = model();
    solver.graph.add_arc(1, 2, -0.2).unwrap();
    matches_fresh(&mut solver, 1e-9);
    let prepared = solver
        .raw_pricing
        .as_ref()
        .unwrap()
        .solver
        .prepared
        .clone()
        .unwrap();
    solver.cuts.cuts.push(RankOneCut {
        weights: vec![0, 1, 1, 0],
        graph_support: Default::default(),
        graph_pricing_support: Default::default(),
        native_kpath_support: None,
        memory: vec![true; 4],
        graph_dependent_memory: false,
        graph_memories: Default::default(),
        memory_kind: MemoryKind::Vertex,
        arc_memory: Default::default(),
        graph_arc_memories: Default::default(),
        arc_weights: Default::default(),
        visit_once: false,
        native_type2: false,
        denominator: 2,
        dual: 0.2,
        class: CutClass::Packing,
        rhs: 1,
        kind: 0,
        id: 0,
        ordered_support: None,
    });
    for dual in [0.2, 0.4, -0.3, 0.] {
        solver.cuts.cuts[0].dual = dual;
        matches_fresh(&mut solver, 1e-9);
        assert!(Arc::ptr_eq(
            &prepared,
            solver
                .raw_pricing
                .as_ref()
                .unwrap()
                .solver
                .prepared
                .as_ref()
                .unwrap()
        ));
    }
    solver.rollback_point_saved = true;
    let time = solver.controls.time_limit;
    solver.controls.time_limit = Duration::ZERO;
    assert!(solver.price_functor_phase(0, 8).is_err());
    assert!(solver.requires_rollback);
    solver.controls.time_limit = time;
    matches_fresh(&mut solver, 1e-9);
    assert!(
        solver
            .raw_pricing
            .as_ref()
            .unwrap()
            .solver
            .rollback_point_saved
    );
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 0.)
        .unwrap();
    assert!(solver.price_functor_phase(0, 8).is_err());
    assert!(solver.requires_rollback);
    solver
        .options
        .set("BALDEShardTimeThresholdInPricing", 20.)
        .unwrap();
    matches_fresh(&mut solver, 1e-9);
}

#[test]
fn presence_dual_updates_reuse_only_raw_preparation_and_match_fresh_searches() {
    use crate::cuts::soft_conflicts::{Mode, PresenceRow, PricingInput};
    let mut solver = model();
    solver
        .set_presence_rows(vec![PricingInput {
            row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
            arc_events: vec![vec![0], vec![1], vec![], vec![]],
            dual: 0.2,
        }])
        .unwrap();
    matches_fresh(&mut solver, 1e-9);
    let prepared = solver
        .raw_pricing
        .as_ref()
        .unwrap()
        .solver
        .prepared
        .clone()
        .unwrap();
    for dual in [-0.3, 0., 0.4] {
        // Retained cached paths and completion labels must still be discarded.
        solver.enumerate(100.).unwrap();
        let mut rows = solver.presence_rows().to_vec();
        rows[0].dual = dual;
        solver.set_presence_rows(rows).unwrap();
        assert!(solver.engine.is_none() && solver.enumeration.is_none());
        let raw = &solver.raw_pricing.as_ref().unwrap().solver;
        assert_eq!(raw.presence_rows(), solver.presence_rows());
        assert!(!raw.engine.as_ref().unwrap().half_ready);
        assert!(!raw.engine.as_ref().unwrap().full_ready);
        matches_fresh(&mut solver, 1e-9);
        assert!(Arc::ptr_eq(
            &prepared,
            solver
                .raw_pricing
                .as_ref()
                .unwrap()
                .solver
                .prepared
                .as_ref()
                .unwrap()
        ));
    }
    let mut invalid = solver.presence_rows().to_vec();
    invalid[0].dual = f64::NAN;
    assert!(solver.set_presence_rows(invalid).is_err());
    assert!(Arc::ptr_eq(
        &prepared,
        solver
            .raw_pricing
            .as_ref()
            .unwrap()
            .solver
            .prepared
            .as_ref()
            .unwrap()
    ));
    let mut changed = solver.presence_rows().to_vec();
    changed[0].arc_events[2].push(0);
    solver.set_presence_rows(changed).unwrap();
    assert!(solver.raw_pricing.is_none());
    matches_fresh(&mut solver, 1e-9);
    assert!(!Arc::ptr_eq(
        &prepared,
        solver
            .raw_pricing
            .as_ref()
            .unwrap()
            .solver
            .prepared
            .as_ref()
            .unwrap()
    ));

    // A topology mutation before a dual-only setter must not be hidden by it.
    solver.graph.arcs[2].cost = 0.07;
    let mut rows = solver.presence_rows().to_vec();
    rows[0].dual = 0.8;
    solver.set_presence_rows(rows).unwrap();
    assert!(solver.raw_pricing.is_none());
    matches_fresh(&mut solver, 1e-9);
}
