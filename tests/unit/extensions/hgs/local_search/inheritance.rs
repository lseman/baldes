use super::*;
use crate::inputs::{Customer, Instance, Rounding};

fn params(n: usize) -> Params {
    let mut instance = Instance {
        name: "inherited-routes".into(),
        vehicles: 3,
        capacity: 2.,
        customers: (0..=n)
            .map(|i| Customer {
                x: 0.,
                y: 0.,
                demand: if i == 0 { 0. } else { 1. },
                ready: 0.,
                due: 1000.,
                service: 0.,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    };
    // Every feasible pair has equal cost. Different complete pairings remain
    // local optima, making inherited and mixed-parent neighborhoods observable.
    instance.distances = Some(
        (0..=n)
            .map(|i| (0..=n).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    );
    let mut params = Params::from_instance(&instance);
    params.config.intensification_probability_ls = 0;
    params.config.nb_granular = n as i32;
    params.penalty_capacity.set(100.);
    params
}

fn educate(engine: &mut HGSLocalSearch<'_>, parent: &mut Individual<'_>) {
    engine.run(parent, 100., 1.);
    assert!(parent.is_feasible);
}

#[test]
fn complete_inherited_routes_skip_ri_pairs_and_keep_the_solution() {
    let params = params(4);
    let mut engine = HGSLocalSearch::new(&params);
    let mut parent = Individual::from_string(&params, "1 2 0 3 4");
    educate(&mut engine, &mut parent);
    let mut baseline = parent.clone();
    engine.run(&mut baseline, 100., 1.);
    let baseline_pairs = engine.ri_pairs_tested;
    let mut child = parent.clone();
    engine.run_with_parents(&mut child, 100., 1., &[&parent]);
    assert!(engine.ri_pairs_reused > 0);
    assert!(engine.ri_pairs_tested < baseline_pairs);
    assert_eq!(child.chrom_r, baseline.chrom_r);
    assert_eq!(
        child.my_cost_sol.penalized_cost,
        baseline.my_cost_sol.penalized_cost
    );
}

#[test]
fn changed_routes_and_moves_reactivate_inherited_pairs() {
    let params = params(4);
    let mut engine = HGSLocalSearch::new(&params);
    let mut parent = Individual::from_string(&params, "1 2 0 3 4");
    educate(&mut engine, &mut parent);
    let child = parent.clone();
    engine.load_individual(&child);
    engine.prepare_inheritance(&child, &[&parent]);
    assert!(engine.reuses_ri_pair(0, 1));
    let first = child.chrom_r[0][0];
    let second = child.chrom_r[1][0];
    engine.swap_node(first, second);
    engine.nb_moves += 1;
    engine.update_route_data(0);
    engine.update_route_data(1);
    assert!(!engine.reuses_ri_pair(0, 1));
    assert!(!engine.reuses_ri_pair(0, 0));
    let mut changed = parent.clone();
    changed.chrom_r[0].reverse();
    engine.load_individual(&changed);
    engine.prepare_inheritance(&changed, &[&parent]);
    assert!(!engine.reuses_ri_pair(0, 1));
    assert!(!engine.reuses_ri_pair(0, 0));
    assert!(engine.reuses_ri_pair(1, 1));
}

#[test]
fn penalty_and_granularity_changes_disable_inheritance() {
    for change in 0..3 {
        let params = params(4);
        let mut engine = HGSLocalSearch::new(&params);
        let mut parent = Individual::from_string(&params, "1 2 0 3 4");
        educate(&mut engine, &mut parent);
        let mut child = parent.clone();
        let (capacity, time_warp) = match change {
            0 => (101., 1.),
            1 => (100., 2.),
            _ => {
                engine.set_granular(3);
                (100., 1.)
            }
        };
        engine.run_with_parents(&mut child, capacity, time_warp, &[&parent]);
        assert_eq!(engine.ri_pairs_reused, 0);
        assert!(engine.ri_pairs_tested > 0);
    }
}

#[test]
fn uneducated_parents_and_evicted_searches_cannot_skip_work() {
    let params = params(4);
    let mut engine = HGSLocalSearch::new(&params);
    engine.searched_plans = inheritance::SearchedPlans::new(1);
    let mut parent = Individual::from_string(&params, "1 2 0 3 4");
    let mut child = parent.clone();
    engine.run_with_parents(&mut child, 100., 1., &[&parent]);
    assert_eq!(engine.ri_pairs_reused, 0);
    educate(&mut engine, &mut parent);
    let mut other = Individual::from_string(&params, "1 3 0 2 4");
    educate(&mut engine, &mut other);
    let mut child = parent.clone();
    engine.run_with_parents(&mut child, 100., 1., &[&parent]);
    assert_eq!(engine.ri_pairs_reused, 0);
    assert!(engine.ri_pairs_tested > 0);
}

#[test]
fn route_order_changes_and_mixed_parents_do_not_certify_pair_searches() {
    let params = params(6);
    let mut engine = HGSLocalSearch::new(&params);
    let mut a = Individual::from_string(&params, "1 2 0 3 4 0 5 6");
    let mut b = Individual::from_string(&params, "1 3 0 2 5 0 4 6");
    educate(&mut engine, &mut a);
    educate(&mut engine, &mut b);
    let mixed = Individual::from_string(&params, "1 2 0 4 6 0 3 5");
    engine.load_individual(&mixed);
    engine.prepare_inheritance(&mixed, &[&a, &b]);
    assert!(engine.reuses_ri_pair(0, 0));
    assert!(engine.reuses_ri_pair(1, 1));
    assert!(!engine.reuses_ri_pair(0, 1));
    assert!(!engine.reuses_ri_pair(2, 2));
    let mut reordered = a.clone();
    reordered.chrom_r.swap(0, 1);
    engine.load_individual(&reordered);
    engine.prepare_inheritance(&reordered, &[&a]);
    assert!(engine.reuses_ri_pair(0, 0));
    assert!(engine.reuses_ri_pair(1, 1));
    assert!(!engine.reuses_ri_pair(0, 1));
}

#[test]
fn route_suffixes_and_prefixes_are_not_complete_inherited_routes() {
    let params = params(4);
    let mut engine = HGSLocalSearch::new(&params);
    let mut parent = Individual::from_string(&params, "1 2 0 3 4");
    educate(&mut engine, &mut parent);
    let child = Individual::from_string(&params, "1 0 2 0 3 4");
    engine.load_individual(&child);
    engine.prepare_inheritance(&child, &[&parent]);
    assert!(!engine.reuses_ri_pair(0, 0));
    assert!(!engine.reuses_ri_pair(1, 1));
    assert!(engine.reuses_ri_pair(2, 2));
}

#[test]
fn infeasible_inherited_routes_are_searched_again() {
    let mut params = params(2);
    params.nb_vehicles = 1;
    params.vehicle_capacity = 1;
    let mut engine = HGSLocalSearch::new(&params);
    let mut parent = Individual::from_string(&params, "1 2");
    engine.run(&mut parent, 100., 1.);
    assert!(!parent.is_feasible);
    let mut child = parent.clone();
    engine.run_with_parents(&mut child, 100., 1., &[&parent]);
    assert_eq!(engine.ri_pairs_reused, 0);
    assert!(engine.ri_pairs_tested > 0);
}
