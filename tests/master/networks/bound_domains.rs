//! Aggregate activity caps retain their mathematical domain at small bounds.
use super::*;
use baldes::master::preprocessing::Bounds;

#[test]
fn activity_caps_do_not_turn_into_pricing_availability_thresholds() {
    let p = triangle();
    let root = mp::solve(p.clone(), options()).unwrap();
    optimal(&root, 1.5);
    let mut arcs = vec![0.; p.networks[0].graph.arcs.len()];
    arcs[0] = 1.;
    let coefficients = [RowCoefficients {
        network: 7,
        arcs,
        per_route: 0.,
    }];
    // Adjacent values straddling a native availability threshold are ordinary
    // positive aggregate caps here. No availability tolerance replaces the row.
    let boundary = 1e-6_f64 / (1. - 1e-10);
    let values = [
        0.,
        0.25,
        1e-6,
        f64::from_bits(boundary.to_bits() - 1),
        boundary,
        f64::from_bits(boundary.to_bits() + 1),
        2e-6,
    ];
    for cap in values {
        let child = root
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: 0.,
                    upper: cap,
                },
                &coefficients,
            )
            .unwrap();
        for integer_domains in [false, true] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                let mut configured = options();
                configured.cg.bound_mode = mode;
                configured.preprocessing.integer_domains = integer_domains;
                let output = child.solve(configured).unwrap();
                let expected = if integer_domains { 2. } else { 2. - cap };
                optimal(&output, expected);
                let arc_activity: f64 = output
                    .routes
                    .iter()
                    .zip(&output.values)
                    .filter(|(route, _)| route.network == 7 && route.route.arcs.contains(&0))
                    .map(|(_, value)| *value)
                    .sum();
                assert!(arc_activity <= cap + 1e-8, "{arc_activity} exceeds {cap}");
                if !integer_domains && cap > 0. {
                    assert!((arc_activity - cap).abs() < 1e-8);
                }
            }
        }
        optimal(&root.snapshot.solve(options()).unwrap(), 1.5);
    }
}

#[test]
fn propagated_zero_activity_and_enumeration_do_not_leak_into_siblings() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut configured = options();
        configured.cg.bound_mode = mode;
        let problem = triangle();
        let root = mp::solve(problem.clone(), configured.clone()).unwrap();
        let mut arcs = vec![0.; problem.networks[0].graph.arcs.len()];
        arcs[0] = 1.;
        let row = [RowCoefficients {
            network: 7,
            arcs,
            per_route: 0.,
        }];
        let parent = root
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: 0.,
                    upper: 1.,
                },
                &row,
            )
            .unwrap()
            .solve(configured.clone())
            .unwrap();
        optimal(&parent, 1.5);
        let child = parent
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: 0.,
                    upper: 0.,
                },
                &row,
            )
            .unwrap()
            .solve(configured.clone())
            .unwrap();
        optimal(&child, 2.);
        let bounds = &child
            .snapshot
            .preprocessing_report()
            .networks
            .iter()
            .find(|n| n.network == 7)
            .unwrap()
            .arcs;
        assert_eq!(bounds[0].upper, 0.);
        let complete = child
            .snapshot
            .enumerate(&mp::EnumerationOptions::default())
            .unwrap()
            .preprocess()
            .unwrap();
        assert!(complete.enumeration_report().complete);
        optimal(&complete.solve(configured.clone()).unwrap(), 2.);
        // A bound on the zero child cannot widen it; the sibling is built from
        // the parent and must price/use the first-pair path again to attain 7/4.
        optimal(
            &child
                .snapshot
                .with_activity_bounds(
                    Bounds {
                        lower: 0.,
                        upper: 1.,
                    },
                    &row,
                )
                .unwrap()
                .solve(configured.clone())
                .unwrap(),
            2.,
        );
        let sibling = parent
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: 0.,
                    upper: 0.25,
                },
                &row,
            )
            .unwrap()
            .solve(configured.clone())
            .unwrap();
        optimal(&sibling, 1.75);
        let activity: f64 = sibling
            .routes
            .iter()
            .zip(&sibling.values)
            .filter(|(route, _)| route.network == 7 && route.route.arcs.contains(&0))
            .map(|(_, v)| *v)
            .sum();
        assert_eq!(activity, 0.25);
        assert!(!sibling.snapshot.enumeration_report().complete);
        optimal(&parent.snapshot.solve(configured.clone()).unwrap(), 1.5);
        optimal(&root.snapshot.solve(configured).unwrap(), 1.5);
    }
}
