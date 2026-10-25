use super::*;
use mp::NetworkPricingSchedule;

#[test]
fn automatic_curvature_without_eligible_rows_does_not_fabricate_feedback() {
    let mut configured = options();
    configured.cg.stabilize = true;
    configured.cg.proximal =
        Some(baldes::master::stabilization::ProximalParameters::automatic_three_piece());
    let root = mp::solve(
        Problem {
            rows: vec![],
            networks: vec![network(7, &[(1., &[])], 0)],
        },
        configured,
    )
    .unwrap();
    optimal(&root, 0.);
    let state = root
        .master
        .warm_start
        .proximal
        .unwrap()
        .snapshot
        .curvature
        .unwrap();
    assert_eq!(state.curvature, 1e12);
    assert_eq!(state.outer_radius, 1e12);
}

#[test]
fn automatic_curvature_initializes_and_survives_resume_and_child_boundaries() {
    use baldes::master::stabilization::ProximalParameters;
    for proximal in [
        ProximalParameters::automatic_three_piece(),
        ProximalParameters::automatic_five_piece(),
    ] {
        let mut configured = options();
        configured.cg.stabilize = true;
        configured.cg.proximal = Some(proximal);
        configured.network_pricing_schedule = NetworkPricingSchedule::NativePromisingFirst;
        let parent = mp::solve(triangle(), configured.clone()).unwrap();
        optimal(&parent, 1.5);
        let automatic = &parent.master.warm_start.proximal.as_ref().unwrap().snapshot;
        let curvature = automatic.curvature.unwrap();
        assert!(curvature.curvature > 0. && curvature.curvature < 1e12);
        let mut stopped = configured.clone();
        stopped.cg.time_limit = std::time::Duration::ZERO;
        let resumed = parent.snapshot.solve(stopped.clone()).unwrap();
        let checkpoint = &resumed
            .master
            .warm_start
            .proximal
            .as_ref()
            .unwrap()
            .snapshot;
        assert_eq!(checkpoint.curvature, Some(curvature));
        assert_eq!(checkpoint.center, automatic.center);
        let child = parent.snapshot.branch_assignment(0, 7, true).unwrap();
        let inherited = child.solve(stopped).unwrap();
        let checkpoint = &inherited
            .master
            .warm_start
            .proximal
            .as_ref()
            .unwrap()
            .snapshot;
        assert_eq!(checkpoint.curvature, Some(curvature));
        assert_eq!(
            &checkpoint.center[..automatic.center.len()],
            automatic.center.as_slice()
        );
        assert!(checkpoint.center[automatic.center.len()..]
            .iter()
            .all(|v| *v == 0.));
        optimal(&child.solve(configured.clone()).unwrap(), 2.);
        let tree = parent
            .snapshot
            .solve_assignment_tree(mp::AssignmentTreeOptions {
                node: configured,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
        assert_eq!(tree.incumbent, 2.);
    }
}

#[test]
fn network_schedules_preserve_roots_children_trees_and_transferred_universes() {
    for policy in [
        NetworkPricingSchedule::FullSweep,
        NetworkPricingSchedule::Cyclic,
        NetworkPricingSchedule::Sorted,
        NetworkPricingSchedule::NativePromisingFirst,
        NetworkPricingSchedule::NativeUnvisitedFirst,
    ] {
        for stabilize in [false, true] {
            let mut configured = options();
            configured.network_pricing_schedule = policy;
            configured.cg.stabilize = stabilize;
            let parent = mp::solve(triangle(), configured.clone()).unwrap();
            optimal(&parent, 1.5);
            let child = parent.snapshot.branch_assignment(0, 7, true).unwrap();
            optimal(&child.solve(configured.clone()).unwrap(), 2.);
            optimal(&parent.snapshot.solve(configured.clone()).unwrap(), 1.5);
            let tree = parent
                .snapshot
                .solve_assignment_tree(mp::AssignmentTreeOptions {
                    node: configured.clone(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
            assert!((tree.incumbent - 2.).abs() < 1e-7);
            configured.enumeration = Some(mp::EnumerationOptions::default());
            let enumerated = parent.snapshot.solve(configured).unwrap();
            optimal(&enumerated, 1.5);
            assert!(enumerated.snapshot.enumeration_report().complete);
        }
    }
}

#[test]
fn explicit_proximal_bands_preserve_original_network_bounds_and_node_domains() {
    use baldes::master::stabilization::{ProximalBand, ProximalParameters};
    let mut bundle = ProximalParameters::three_piece(0.5, 1.);
    bundle.multi_point = true;
    for proximal in [
        ProximalParameters::box_step(0.5),
        ProximalParameters::three_piece(0.5, 1.),
        ProximalParameters::five_piece(
            ProximalBand {
                half_width: 0.1,
                penalty: 0.5,
            },
            ProximalBand {
                half_width: 0.5,
                penalty: 1.,
            },
        ),
        bundle,
    ] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let mut configured = options();
            configured.cg.stabilize = true;
            configured.cg.proximal = Some(proximal.clone());
            configured.cg.bound_mode = mode;
            configured.network_pricing_schedule = NetworkPricingSchedule::NativePromisingFirst;
            let root = mp::solve(triangle(), configured.clone()).unwrap();
            optimal(&root, 1.5);
            let child = root.snapshot.branch_assignment(0, 7, true).unwrap();
            optimal(&child.solve(configured.clone()).unwrap(), 2.);
            let tree = root
                .snapshot
                .solve_assignment_tree(mp::AssignmentTreeOptions {
                    node: configured,
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(tree.status, baldes::master::tree::TreeStatus::Optimal);
            assert_eq!(tree.incumbent, 2.);
        }
    }
}

#[test]
fn one_partial_exact_batch_cannot_publish_a_bound_above_an_unvisited_network() {
    for policy in [
        NetworkPricingSchedule::Cyclic,
        NetworkPricingSchedule::Sorted,
    ] {
        let problem = Problem {
            rows: vec![],
            networks: vec![
                network(7, &[(-1., &[])], 0),
                network(42, &[(-100., &[])], 0),
            ],
        };
        let mut configured = options();
        configured.preprocessing.enabled = false;
        configured.network_pricing_schedule = policy;
        configured.cg.max_iterations = 1;
        let limited = mp::solve(problem.clone(), configured.clone()).unwrap();
        assert_eq!(limited.oracle_calls, vec![(7, 1), (42, 0)]);
        assert_eq!(limited.master.status, Status::IterationLimit);
        assert!(!limited.master.history[0].exact);
        assert_eq!(limited.master.numerical_bound, f64::NEG_INFINITY);
        configured.cg.max_iterations = 100;
        let full = mp::solve(problem, configured).unwrap();
        optimal(&full, -303.);
    }
}
