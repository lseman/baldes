use super::*;
use crate::cuts::{CutClass, MemoryKind, RankOneCut};
use std::collections::{BTreeMap, BTreeSet};

fn solver() -> Solver<2> {
    let mut solver = Solver::<2>::new(6, 0).unwrap();
    for v in 0..5 {
        solver.graph.add_arc(v, v + 1, -2.).unwrap();
    }
    solver.cuts.cuts.push(RankOneCut {
        weights: vec![0, 1, 1, 0, 0, 0],
        graph_support: BTreeMap::new(),
        graph_pricing_support: BTreeMap::new(),
        native_kpath_support: None,
        memory: vec![true; 6],
        graph_dependent_memory: false,
        graph_memories: BTreeMap::new(),
        memory_kind: MemoryKind::Vertex,
        arc_memory: BTreeSet::new(),
        graph_arc_memories: BTreeMap::new(),
        arc_weights: BTreeMap::new(),
        visit_once: false,
        native_type2: false,
        denominator: 2,
        dual: -1.,
        class: CutClass::Packing,
        rhs: 1,
        kind: 0,
        id: 0,
        ordered_support: Some(vec![(1, 1), (2, 1)]),
    });
    solver
}

#[test]
fn dual_only_refresh_reuses_source_and_engine_inputs_and_matches_fresh_pricing() {
    for enumerated in [false, true] {
        let mut cached = solver();
        cached.price_phase(0, 4, false).unwrap();
        if enumerated {
            cached.enumerate(0.).unwrap();
        }
        let source = cached.cut_input_source.as_ref().unwrap().clone();
        let sibling = cached.fork_for_node();
        assert!(Arc::ptr_eq(
            &source,
            sibling.cut_input_source.as_ref().unwrap()
        ));
        let coefficients = cached.applied_cuts.0[0].vertex_coefficient.as_ptr();
        let engine_coefficients = cached.engine.as_ref().unwrap().cuts.inputs[0]
            .vertex_coefficient
            .as_ptr();
        cached.cuts.cuts[0].dual = -2.;
        cached.refresh_cuts().unwrap();
        assert!(Arc::ptr_eq(
            &source,
            cached.cut_input_source.as_ref().unwrap()
        ));
        assert_eq!(
            cached.applied_cuts.0[0].vertex_coefficient.as_ptr(),
            coefficients
        );
        assert_eq!(
            cached.engine.as_ref().unwrap().cuts.inputs[0]
                .vertex_coefficient
                .as_ptr(),
            engine_coefficients
        );
        let mut fresh = cached.fork_for_node();
        fresh.cut_input_source = None;
        fresh.applied_cuts = (vec![], vec![], vec![], vec![]);
        fresh.engine = None;
        fresh.prepared = None;
        fresh.enumeration = None;
        fresh.enumeration_snapshot = None;
        let expected = fresh.price_phase(0, 4, false).unwrap();
        let actual = cached.price_phase(0, 4, false).unwrap();
        assert_eq!(
            actual.minimum_reduced_cost.to_bits(),
            expected.minimum_reduced_cost.to_bits()
        );
        assert_eq!(actual.columns.len(), expected.columns.len());
        for (a, b) in actual.columns.iter().zip(&expected.columns) {
            assert_eq!(a.nodes, b.nodes);
            assert_eq!(a.arcs, b.arcs);
            assert_eq!(a.cost.to_bits(), b.cost.to_bits());
        }
    }
}

#[test]
fn rebuilt_dual_inputs_reuse_engine_arrays_and_reload_mismatched_layouts() {
    for mismatched_engine in [false, true] {
        let mut cached = solver();
        cached.price_phase(0, 4, false).unwrap();
        let coefficients = cached.engine.as_ref().unwrap().cuts.inputs[0]
            .vertex_coefficient
            .as_ptr();
        if mismatched_engine {
            cached.engine.as_mut().unwrap().cuts.inputs[0]
                .memory_vertices
                .clear();
        }
        cached.cut_input_source = None; // Force the input-rebuild fallback.
        cached.cuts.cuts[0].dual = -2.;
        cached.refresh_cuts().unwrap();
        let engine = cached.engine.as_ref().unwrap();
        assert!(engine.cut_structure_matches(&cached.applied_cuts.0, &cached.applied_cuts.1));
        assert_eq!(engine.cuts.inputs[0].dual, cached.applied_cuts.0[0].dual);
        if !mismatched_engine {
            assert_eq!(
                engine.cuts.inputs[0].vertex_coefficient.as_ptr(),
                coefficients
            );
        }
        let mut fresh = cached.fork_for_node();
        fresh.cut_input_source = None;
        fresh.applied_cuts = (vec![], vec![], vec![], vec![]);
        fresh.engine = None;
        fresh.prepared = None;
        fresh.enumeration = None;
        fresh.enumeration_snapshot = None;
        let expected = fresh.price_phase(0, 4, false).unwrap();
        let actual = cached.price_phase(0, 4, false).unwrap();
        assert_eq!(
            actual.minimum_reduced_cost.to_bits(),
            expected.minimum_reduced_cost.to_bits()
        );
        assert_eq!(actual.columns.len(), expected.columns.len());
        for (a, b) in actual.columns.iter().zip(&expected.columns) {
            assert_eq!(a.nodes, b.nodes);
            assert_eq!(a.arcs, b.arcs);
            assert_eq!(a.cost.to_bits(), b.cost.to_bits());
        }
    }
}
