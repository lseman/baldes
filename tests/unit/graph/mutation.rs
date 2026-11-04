use super::*;

#[test]
fn invalid_arc_edit_rolls_back_and_cost_edit_updates_both_objectives() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    let id = graph.add_arc(0, 1, 2.).unwrap();
    let original = graph.clone();
    assert!(graph.edit_arc(id, |arc| arc.head = 20).is_err());
    assert_eq!(graph, original);
    assert!(graph.set_arc_cost(id, f64::NAN).is_err());
    assert_eq!(graph, original);
    graph.set_arc_cost(id, 3.).unwrap();
    assert_eq!(graph.arcs()[0].cost, 3.);
    assert_eq!(graph.arcs()[0].reduced_cost, 3.);
    assert_eq!(graph.vertices().len(), 2);
    assert!(graph.resources().is_empty());
}

#[test]
fn revisions_distinguish_independent_clone_edits_and_ignore_objective_refresh() {
    let mut original = Graph::<2>::new(2, 0).unwrap();
    original.add_arc(0, 1, 2.).unwrap();
    let mut first = original.clone();
    let mut second = original.clone();
    let revision = original.revision;
    first.set_reduced_costs(&[-3.]).unwrap();
    assert_eq!(first.revision, revision);
    assert!(first.same_topology(&original));
    assert_ne!(first, original);
    assert!(first.set_reduced_costs(&[f64::NAN]).is_err());
    assert_eq!(first.arcs[0].reduced_cost, -3.);
    first.edit().arcs[0].cost = 4.;
    second.edit().arcs[0].cost = 5.;
    assert_ne!(first.revision, second.revision);
    assert!(!first.same_topology(&second));
    assert!(!first.same_topology(&original));
    // A conservative revision bump still allows value-equivalent graphs.
    second.edit().arcs[0].cost = 4.;
    assert!(first.same_topology(&second));
    assert_eq!(original.arcs[0].cost, 2.);
}
