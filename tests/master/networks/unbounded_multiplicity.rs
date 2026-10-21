use super::*;
use baldes::master::{preprocessing::Bounds, tree::TreeStatus};
use multi_pricing::{AssignmentTreeOptions, EnumerationOptions};

fn unbounded(mut p: Problem<2>) -> Problem<2> {
    for n in &mut p.networks {
        n.multiplicity.upper = f64::INFINITY;
    }
    p
}
#[test]
fn shared_rows_bound_unlimited_networks_through_fixing_enumeration_and_tree() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let root = multi_pricing::solve(unbounded(triangle()), configured.clone()).unwrap();
        optimal(&root, 1.5);
        assert!(root
            .snapshot
            .preprocessing_report()
            .networks
            .iter()
            .all(|n| n.multiplicity.upper.is_finite()));
        let tree = root
            .snapshot
            .solve_assignment_tree(AssignmentTreeOptions {
                node: configured.clone(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 2.);
        let tag = root
            .routes
            .iter()
            .position(|r| r.network == 7 && r.route.arcs == [0, 1])
            .unwrap();
        let fixed = root.snapshot.fix_route(tag, 1.).unwrap();
        optimal(&fixed.solve(configured.clone()).unwrap(), 2.);
        let complete = fixed.enumerate(&EnumerationOptions::default()).unwrap();
        assert!(complete.enumeration_report().complete);
        optimal(&complete.solve(configured.clone()).unwrap(), 2.);
        #[cfg(feature = "highs")]
        assert_eq!(
            complete
                .solve_enumerated_mip(&Default::default())
                .unwrap()
                .unwrap()
                .mip
                .solution
                .unwrap()
                .objective,
            2.
        );
        let disabled = root
            .snapshot
            .with_multiplicity(
                7,
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
            )
            .unwrap();
        optimal(&disabled.solve(configured.clone()).unwrap(), 2.);
        optimal(&root.snapshot.solve(configured).unwrap(), 1.5);
    }
}
#[test]
fn unlimited_multiplicity_with_a_positive_lower_bound_retains_original_rows() {
    let mut p = unbounded(Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![network(7, &[(1., &[0])], 1)],
    });
    p.networks[0].multiplicity.lower = 1.;
    let mut certified = options();
    certified.cg.bound_mode = BoundMode::Certified;
    let root = multi_pricing::solve(p, certified.clone()).unwrap();
    optimal(&root, 1.);
    let fixed = root.snapshot.fix_route(0, 1.).unwrap();
    optimal(&fixed.solve(certified).unwrap(), 1.);
}
#[test]
fn truly_unbounded_route_directions_and_neutral_infinite_boxes_are_distinct() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let p = unbounded(Problem {
            rows: vec![],
            networks: vec![network(7, &[(-1., &[])], 0)],
        });
        let root = multi_pricing::solve(p, configured.clone()).unwrap();
        assert_eq!(root.master.status, Status::Unbounded);
        assert_eq!(root.master.bound, f64::NEG_INFINITY);
        let p = unbounded(Problem {
            rows: vec![],
            networks: vec![network(7, &[(0., &[])], 0)],
        });
        let root = multi_pricing::solve(p, configured.clone()).unwrap();
        optimal(&root, 0.);
        let full = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        optimal(&full.solve(configured).unwrap(), 0.);
    }
}
#[test]
fn shared_route_quantities_can_exceed_the_legacy_backend_default_cap() {
    let quantity = 2_000_000_000_001.;
    for unlimited in [false, true] {
        let mut p = Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: quantity,
            }],
            networks: vec![network(7, &[(1., &[0])], 1)],
        };
        p.networks[0].multiplicity.upper = if unlimited { f64::INFINITY } else { quantity };
        let mut configured = options();
        configured.cg.bound_mode = BoundMode::Certified;
        let root = multi_pricing::solve(p, configured.clone()).unwrap();
        optimal(&root, quantity);
        assert_eq!(root.values.iter().sum::<f64>(), quantity);
        let fixed = root.snapshot.fix_route(0, quantity).unwrap();
        optimal(&fixed.solve(configured).unwrap(), quantity);
    }
}
#[test]
fn invalid_network_infinities_are_rejected() {
    for (lower, upper) in [
        (f64::INFINITY, f64::INFINITY),
        (0., f64::NEG_INFINITY),
        (0., f64::NAN),
        (-1., f64::INFINITY),
    ] {
        let mut p = triangle();
        p.networks[0].multiplicity = Window { lower, upper };
        assert!(multi_pricing::solve(p, options()).is_err());
    }
}
