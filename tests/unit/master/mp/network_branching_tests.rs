use super::*;

#[test]
fn multiplicity_first_matches_the_mapped_native_first_child_and_restores_parent() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    let root = solve_with_master_variables(problem, vec![], options.clone()).unwrap();
    assert_eq!(root.master.lp, 1.5);
    let branches = children(&root, 1e-6, NetworkBranchingPolicy::MultiplicityFirst)
        .unwrap()
        .unwrap();
    assert_eq!(branches.len(), 2);
    let first = branches[0].solve(options.clone()).unwrap();
    // In the public native control, fractional network42 count .5 splits
    // upward first, and requiring one copy raises the exact LP to5/3.
    assert!(
        (first.master.lp - 5. / 3.).abs() < 1e-8,
        "{:?}",
        first.master
    );
    let quantity: f64 = first
        .routes
        .iter()
        .zip(&first.values)
        .filter(|(route, _)| route.network == 42)
        .map(|(_, value)| value)
        .sum();
    assert!((quantity - 1.).abs() < 1e-8);
    assert_eq!(root.snapshot.solve(options).unwrap().master.lp, 1.5);
}
