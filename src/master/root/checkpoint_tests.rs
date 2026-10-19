use super::*;
use std::time::Duration;

fn controller() -> (RootController, ClpMaster, Output) {
    let instance = crate::vrptw::Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
    let problem =
        RootProblem::set_partitioning(instance.graph(1, crate::vrptw::Rounding::None).unwrap(), 3.)
            .unwrap();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.cg.stabilize = false;
    let solved = solve(problem, options).unwrap();
    let snapshot = solved.snapshot.unwrap();
    let master = snapshot.master.try_clone().unwrap();
    let state = State {
        solver: snapshot.pricing.fork_for_node(),
        routes: snapshot.routes.clone(),
        route_tags: snapshot
            .routes
            .iter()
            .enumerate()
            .map(|(i, r)| (r.identity(), i as i32))
            .collect(),
        incumbent: snapshot.incumbent,
        enumerate: false,
        rounded: false,
        previous_fixing_gap: 1e12,
        deadline: None,
        pricing_seconds: 0.,
        cut: snapshot.cut.clone(),
        fixing_calls: 0,
        enumerated_routes: 0,
    };
    (
        RootController {
            problem: Rc::new(snapshot.problem.clone()),
            state: Rc::new(RefCell::new(state)),
            checkpoint: None,
        },
        master,
        solved.master,
    )
}

#[test]
fn child_penalties_restart_without_changing_parent_or_real_column_costs() {
    let (mut controller, master, output) = controller();
    controller
        .save_pricing_checkpoint(&master, &output)
        .unwrap();
    let mut parent = Rc::try_unwrap(controller.checkpoint.take().unwrap().snapshot).unwrap();
    assert!(!parent.warm.artificials.is_empty());
    for &column in &parent.warm.artificials {
        parent.master.set_cost(column, 88.).unwrap();
    }
    let parent_costs = parent.master.costs.clone();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.cg.artificial_cost = 3.;
    options.cg.time_limit = Duration::ZERO;
    let child = inherit_branches(&parent, &[], options).unwrap();
    let child = child.snapshot.unwrap();
    for (column, &cost) in child.master.costs.iter().enumerate() {
        let expected = if parent.warm.artificials.contains(&column) {
            3.
        } else {
            parent_costs[column]
        };
        assert_eq!(cost, expected);
    }
    assert_eq!(parent.master.costs, parent_costs);
}

#[test]
fn pricing_timeout_restores_complete_pre_cut_state_and_preserves_incumbent() {
    let (mut controller, mut master, output) = controller();
    controller
        .save_pricing_checkpoint(&master, &output)
        .unwrap();
    let checkpoint = controller.checkpoint.as_ref().unwrap().snapshot.clone();
    let saved_rows = master.rhs.clone();
    let saved_values = master.values();
    let saved_routes = controller.state.borrow().routes.len();
    master.add_row(RowSense::LessEqual, 0., &[]).unwrap();
    {
        let mut state = controller.state.borrow_mut();
        state.incumbent = 4. + 2f64.sqrt();
        state.cut.rounds += 1;
        state.previous_fixing_gap = 0.;
        state.enumerated_routes = 999;
        state.routes.clear();
        state.route_tags.clear();
        state.solver.controls.time_limit = Duration::ZERO;
    }
    controller.problem.row_map.borrow_mut().reverse();
    let error = controller
        .state
        .borrow_mut()
        .solver
        .price_phase(0, 150, false)
        .unwrap_err();
    assert!(controller.state.borrow().solver.requires_rollback);
    let (restored, recovered) = controller.recover_pricing_timeout(error).unwrap();
    assert_eq!(restored.rhs, saved_rows);
    assert_eq!(restored.values(), saved_values);
    assert_eq!(recovered.bound, output.bound);
    assert!(recovered.bound <= 4. + 2f64.sqrt());
    assert_eq!(recovered.warm_start.columns.len(), restored.costs.len());
    assert_eq!(
        *controller.problem.row_map.borrow(),
        *checkpoint.problem.row_map.borrow()
    );
    let state = controller.state.borrow();
    assert_eq!(state.routes.len(), saved_routes);
    assert_eq!(state.previous_fixing_gap, 1e12);
    assert_eq!(state.enumerated_routes, 0);
    assert_eq!(state.route_tags.len(), saved_routes);
    assert_eq!(state.cut.rounds, checkpoint.cut.rounds);
    assert_eq!(state.incumbent, 4. + 2f64.sqrt());
    assert!(!state.solver.requires_rollback);
    assert!(!state.solver.rollback_point_saved);
    assert!(!state.solver.controls.time_limit.is_zero());
    assert_eq!(checkpoint.master.rhs, saved_rows);
    assert_eq!(checkpoint.master.values(), saved_values);
}

#[test]
fn checkpoint_does_not_swallow_unsaved_timeouts_or_other_failures() {
    let (mut controller, master, output) = controller();
    assert!(controller
        .recover_pricing_timeout(crate::incomplete("pricing time limit"))
        .is_err());
    controller
        .save_pricing_checkpoint(&master, &output)
        .unwrap();
    assert!(controller
        .recover_pricing_timeout(crate::incomplete("pricing time limit"))
        .is_err());
    controller.state.borrow_mut().solver.requires_rollback = true;
    assert!(controller
        .recover_pricing_timeout(crate::incomplete("label limit"))
        .is_err());
    assert!(controller
        .recover_pricing_timeout(crate::invalid("pricing time limit"))
        .is_err());
    assert!(controller.checkpoint.is_some());
}

#[test]
fn real_cut_round_dispatch_saves_a_pricing_checkpoint() {
    let (controller, _, _) = controller();
    let mut options = RootOptions::default();
    options.capture_state = true;
    options.cg.stabilize = false;
    options.cuts.enabled = true;
    let result = solve((*controller.problem).clone(), options).unwrap();
    assert!(result.cuts > 0);
    assert!(result.master.cut_rounds > 0);
    assert!(
        result
            .snapshot
            .as_ref()
            .unwrap()
            .pricing
            .rollback_point_saved
    );
    assert!(!result.snapshot.as_ref().unwrap().pricing.requires_rollback);
    assert!(result.master.bound <= 4. + 2f64.sqrt() + 1e-9);
    assert!((result.master.lp - (4. + 2f64.sqrt())).abs() < 1e-9);
}
