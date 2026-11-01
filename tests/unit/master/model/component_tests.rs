use super::*;
use crate::master::preprocessing::Bounds;
#[cfg(feature = "highs")]
use crate::master::solvers::mip::MipOptions;

fn fixture() -> (Output<2>, Options, ComponentClassBranching) {
    fixture_with_coefficient(1.)
}

fn fixture_with_coefficient(
    cheap_coefficient: f64,
) -> (Output<2>, Options, ComponentClassBranching) {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (a, b, c) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 5.), (2, 3, 0.)] {
        graph.add_arc(a, b, c).unwrap();
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, cheap_coefficient)], vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    options.gap_enumeration = false;
    let class = ComponentClassBranching {
        name: "expensive".into(),
        priority: 1.,
        minimum_fractionality: 0.02,
        network: 7,
        coordinates: vec![ComponentCoordinate {
            id: 99,
            arc_values: vec![None, None, Some(1), None],
        }],
        sequence: ComponentSequence {
            bounds: vec![ComponentBound {
                coordinate: 99,
                threshold: 1.,
                sense: ComponentSense::GreaterEqual,
            }],
        },
    };
    (solve(problem, options.clone()).unwrap(), options, class)
}

#[test]
fn component_class_prices_future_columns_and_replays_fixed_and_enumerated_routes() {
    let (root, options, class) = fixture();
    assert_eq!(root.master.lp, 1.);
    let child = root
        .snapshot
        .with_component_bounds(
            &class,
            Bounds {
                lower: 1.,
                upper: f64::INFINITY,
            },
        )
        .unwrap();
    let solved = child.solve(options.clone()).unwrap();
    assert_eq!(solved.master.lp, 5.);
    assert!(solved.master.certified_bound <= 5.);
    // A degenerate LP basis may put the price on a loaded route's upper cap.
    // At this explicit dual point, joint class/path replay must recover 5;
    // independent [0,1] class envelopes alone yield only 1.
    let mut duals = vec![0.; solved.snapshot.master.rhs.len()];
    let offset = solved.snapshot.state.shared_offset;
    duals[offset] = 1.;
    duals[offset + 1] = 4.;
    let bound =
        bounds::certificate(&solved.snapshot.state, &solved.snapshot.master, &duals).unwrap();
    assert!(bound >= 5. - 1e-7 && bound <= 5., "{bound}");
    let tag = solved
        .routes
        .iter()
        .position(|r| r.route.arcs == [2, 3])
        .unwrap();
    let fixed = solved.snapshot.fix_route(tag, 1.).unwrap();
    assert_eq!(fixed.solve(options.clone()).unwrap().master.lp, 5.);
    let complete = child.enumerate(&EnumerationOptions::default()).unwrap();
    let complete = complete.solve(options).unwrap();
    assert_eq!(complete.master.lp, 5.);
    #[cfg(feature = "highs")]
    assert_eq!(
        complete
            .snapshot
            .solve_enumerated_mip(&MipOptions::default())
            .unwrap()
            .unwrap()
            .mip
            .solution
            .unwrap()
            .objective,
        5.
    );
    assert_eq!(
        root.snapshot.solve(Options::default()).unwrap().master.lp,
        1.
    );
}

#[test]
fn explicit_component_generator_branches_on_class_quantity() {
    let (root, options, mut class) = fixture_with_coefficient(2.);
    assert_eq!(root.master.lp, 0.5);
    class.coordinates[0].arc_values = vec![Some(1), None, None, None];
    class.priority = 1e9;
    let proposal = class.proposal(&root, 1e-7).unwrap().unwrap();
    assert_eq!(proposal.value, 0.5);
    assert!(matches!(
        proposal.restriction,
        model_strong::Restriction::Component(_)
    ));
    let mut tree = ModelTreeOptions::default();
    tree.node = options;
    tree.branching = NetworkBranchingPolicy::Phased;
    tree.component_classes = vec![class];
    let solved = root.snapshot.solve_tree(tree).unwrap();
    assert_eq!(solved.incumbent, 5.);
}
