use super::*;

#[test]
fn adverse_unbounded_static_duals_stay_unknown_while_pricing_can_continue() {
    let problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    let variable = MasterVariable {
        id: 1,
        cost: 0.25,
        bounds: Bounds {
            lower: 0.,
            upper: f64::INFINITY,
        },
        integer: false,
        entries: vec![],
    };
    let mut root = solve_with_master_variables(problem, vec![variable], options).unwrap();
    assert_eq!(root.snapshot.master.upper[0], f64::MAX);
    assert_eq!(
        root.snapshot.state.master_variables[0].bounds.upper,
        f64::INFINITY
    );
    let mut duals = vec![0.; root.snapshot.state.rhs.len()];
    let row = root.snapshot.state.master_variables[0].entries[0].0;
    duals[root.snapshot.state.shared_offset + row] = 1.;
    assert!(correction(&root.snapshot.state, &duals).is_none());
    assert!(
        super::super::bounds::certificate(&root.snapshot.state, &root.snapshot.master, &duals)
            .is_none()
    );
    let phase = root.snapshot.state.oracles[0]
        .solver
        .phases
        .iter()
        .position(|p| p.exact())
        .unwrap();
    let answer = root.snapshot.state.price(phase, &duals).unwrap();
    assert!(answer.exact);
    assert!(root.snapshot.state.correction.is_none());
    assert!(root.snapshot.state.gradient.is_none());
}
