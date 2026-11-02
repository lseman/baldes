use super::*;

#[test]
fn family_registration_order_file_reaches_typed_network_tree_options() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/unit/bin/baldes/cli/family_registration_order.json"
    );
    let parsed = parse_options(vec![
        "--tree".into(),
        "--branching-registrations".into(),
        path.into(),
    ])
    .unwrap()
    .unwrap();
    assert!(matches!(
        parsed.options.branching_registrations.as_slice(),
        [model::BranchingRegistration::FamilyOrder(
            model::BranchFamilyRegistration::PackingResource
        )]
    ));
    let configured = parsed.options.tree_options().unwrap();
    assert_eq!(
        configured.branching_registry.family_order(),
        &[model::BranchFamilyRegistration::PackingResource]
    );
    let mut root = parsed.options;
    root.tree = false;
    assert!(root.node_options().is_err());
}
