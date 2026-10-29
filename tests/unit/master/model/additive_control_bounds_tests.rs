use super::*;

#[test]
fn attained_source_bounds_require_a_witness_and_preserve_budget_unknowns() {
    let source = include_str!("../../../fixtures/coupled/paired-cyclic-active-row-model.json");
    let problem = crate::master::model::format::Model::from_json(source)
        .unwrap()
        .problem::<2>()
        .unwrap();
    let mut options = crate::master::model::Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    let mut output = crate::master::model::solve(problem, options).unwrap();
    let state = &mut output.snapshot.state;
    let oracle = &mut state.oracles[0];
    let duals = vec![0.; state.rhs.len()];
    let deadline = Instant::now() + std::time::Duration::from_secs(1);
    let mut work = 50_000;
    // At zero row prices the independently feasible direct path has original
    // objective -20+1+1. The active shared row is dualized during pricing.
    assert_eq!(
        attained_minimum(oracle, state.shared_offset, &duals, &mut work, deadline),
        Some(Some(-18.))
    );
    let mut no_work = 0;
    assert_eq!(
        attained_minimum(oracle, state.shared_offset, &duals, &mut no_work, deadline),
        None
    );
    let expired = Instant::now() - std::time::Duration::from_secs(1);
    assert_eq!(
        attained_minimum(oracle, state.shared_offset, &duals, &mut work, expired),
        None
    );
    let fractional = oracle;
    for arc in &mut Arc::make_mut(&mut fractional.network.graph).arcs {
        if arc.cost == -20. {
            arc.cost = -20.125;
        }
    }
    assert_eq!(
        attained_minimum(fractional, state.shared_offset, &duals, &mut work, deadline),
        Some(Some(-18.125))
    );
    let mut prices = duals.clone();
    prices[state.shared_offset] = 0.125;
    assert!(prepare(
        fractional,
        state.shared_offset,
        &prices,
        &mut work,
        deadline
    )
    .is_some());
    prices[state.shared_offset] = 0.1;
    assert!(prepare(
        fractional,
        state.shared_offset,
        &prices,
        &mut work,
        deadline
    )
    .is_none());
    let mut definitions = fractional
        .network
        .graph
        .coupled_kernel()
        .unwrap()
        .definitions()
        .to_vec();
    let scale = |function: &crate::graph::coupled::PreparedSlopeFunction| {
        crate::graph::coupled::PreparedSlopeFunction::from_prepared_pairs(
            function.pairs().iter().map(|&(y, x)| (y / 2., x)).collect(),
        )
        .unwrap()
    };
    if let CoupledResource::SlopePaired { initial, arcs, .. } = &mut definitions[0] {
        *initial = scale(initial);
        for arc in arcs {
            arc.increment = scale(&arc.increment);
            arc.lower = scale(&arc.lower);
            arc.upper = scale(&arc.upper);
        }
    } else {
        panic!("expected source slope program");
    }
    Arc::make_mut(&mut fractional.network.graph)
        .set_coupled_resources(&definitions)
        .unwrap();
    // The two-arc source witness now charges .5+.5, with the same physical
    // entry cost -20.125. Both original and redistributed arithmetic are exact.
    assert_eq!(
        attained_minimum(fractional, state.shared_offset, &duals, &mut work, deadline),
        Some(Some(-19.125))
    );
    if let CoupledResource::SlopePaired { arcs, .. } = &mut definitions[0] {
        for arc in arcs {
            let mut pairs = arc.increment.pairs().to_vec();
            pairs[0].0 = 0.1;
            arc.increment =
                crate::graph::coupled::PreparedSlopeFunction::from_prepared_pairs(pairs).unwrap();
        }
    }
    Arc::make_mut(&mut fractional.network.graph)
        .set_coupled_resources(&definitions)
        .unwrap();
    assert!(prepare(fractional, state.shared_offset, &duals, &mut work, deadline).is_none());
}
