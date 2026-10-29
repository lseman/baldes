use super::*;

fn node() -> NodeSnapshot<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut options = Options::default();
    options.cg.time_limit = std::time::Duration::ZERO;
    options.preprocessing.enabled = false;
    solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]; 2],
                route_entries: vec![],
                activity_entries: vec![],
            }],
        },
        options,
    )
    .unwrap()
    .snapshot
}

#[test]
fn explicit_dynamic_rows_and_ordinary_children_keep_physical_classification_independent() {
    let parent = node();
    let original_rows = parent.master.rhs.len();
    assert!(parent
        .state
        .curvature_rows
        .iter()
        .all(|r| r.active && r.class == cg::CurvatureRowClass::Excluded));
    let dynamic = parent
        .with_dynamic_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 10.,
            },
            &[],
        )
        .unwrap();
    let ordinary = dynamic
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.,
            },
            &[],
        )
        .unwrap();
    assert_eq!(parent.state.curvature_rows.len(), original_rows);
    assert_eq!(dynamic.state.curvature_rows.len(), original_rows + 1);
    assert_eq!(
        ordinary.state.curvature_rows[original_rows].class,
        cg::CurvatureRowClass::Dynamic
    );
    assert_eq!(
        ordinary.state.curvature_rows[original_rows + 1].class,
        cg::CurvatureRowClass::Ordinary
    );
    assert!(ordinary.warm.proximal.is_none());
    let count = ordinary.master.rhs.len();
    let controller = Coordinator(Rc::new(RefCell::new(ordinary.state)), false, false, None);
    let rows = cg::Controller::<ClpMaster>::curvature_rows(&controller, count + 1);
    assert_eq!(rows[original_rows].class, cg::CurvatureRowClass::Dynamic);
    assert_eq!(
        rows[original_rows + 1].class,
        cg::CurvatureRowClass::Ordinary
    );
    assert_eq!(rows[count].class, cg::CurvatureRowClass::Excluded);
    assert!(!rows[count].active);
}

#[test]
fn plain_fork_preserves_row_roles_without_sharing_mutable_classification() {
    let parent = node();
    let mut child = parent.fork().unwrap();
    child.state.curvature_rows[0].class = cg::CurvatureRowClass::Dynamic;
    assert_eq!(
        parent.state.curvature_rows[0].class,
        cg::CurvatureRowClass::Excluded
    );
    assert!(!child.state.scheduler.complete);
}
