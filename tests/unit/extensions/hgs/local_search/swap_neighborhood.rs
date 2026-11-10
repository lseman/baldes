use super::*;
use crate::extensions::hgs::{local_search::HGSLocalSearch, Individual};
use crate::inputs::{Customer, Instance, Rounding};

fn params(n: usize) -> Params {
    let instance = Instance {
        name: "swap-filter".into(),
        vehicles: 2,
        capacity: 100.,
        customers: (0..=n)
            .map(|i| Customer {
                x: i as f64,
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
    let mut params = Params::from_instance(&instance);
    params.config.intensification_probability_ls = 100;
    params.config.swap_star_neighbors = 1;
    params.config.swap_star_demand_percentage = 50;
    params.config.swap_star_full_pass_interval = 3;
    params
}

#[test]
fn demand_prefilter_precedes_distance_and_is_independent_of_ri_granularity() {
    let mut params = params(5);
    for (i, demand) in [0, 10, 11, 12, 100, 100].into_iter().enumerate() {
        params.cli[i].demand = demand;
        for j in 0..=5 {
            params.time_cost.set(i, j, if i == j { 0. } else { 100. });
        }
    }
    for (i, j, cost) in [
        (1, 2, 20.),
        (1, 3, 2.),
        (1, 4, 0.25),
        (2, 3, 1.),
        (4, 5, 1.),
        (5, 4, 1.),
    ] {
        params.time_cost.set(i, j, cost);
    }
    let first = SwapNeighborhood::new(&params);
    assert_eq!(first.neighbors[1], vec![3]);
    params.config.nb_granular = 1;
    let second = SwapNeighborhood::new(&params);
    assert_eq!(first.neighbors, second.neighbors);
    for (i, list) in first.neighbors.iter().enumerate() {
        for &j in list {
            assert_ne!(i, j);
            assert_ne!(j, 0);
            assert!(first.neighbors[j].binary_search(&i).is_ok());
        }
    }
}

#[test]
fn periodic_broad_passes_refresh_filtered_timestamps_and_survive_run_resets() {
    let params = params(5);
    let mut neighborhood = SwapNeighborhood::new(&params);
    let mut broad = Vec::new();
    for i in 0..7 {
        let refresh = neighborhood.begin_pass();
        broad.push(neighborhood.broad);
        assert_eq!(refresh, i == 3 || i == 6);
    }
    assert_eq!(broad, vec![true, false, false, true, false, false, true]);
    neighborhood.begin_pass();
    assert!(!neighborhood.broad);
    neighborhood.reset_run();
    neighborhood.begin_pass();
    assert!(!neighborhood.broad); // cadence did not restart at every LS call
    assert!(neighborhood.begin_pass());
    assert!(neighborhood.broad);
}

#[test]
fn disabling_filtering_preserves_broad_passes_without_division_by_zero() {
    for (neighbors, interval) in [(0, 5), (35, 0), (35, 1)] {
        let mut params = params(4);
        params.config.swap_star_neighbors = neighbors;
        params.config.swap_star_full_pass_interval = interval;
        let mut neighborhood = SwapNeighborhood::new(&params);
        for _ in 0..8 {
            assert!(!neighborhood.begin_pass());
            assert!(neighborhood.broad);
        }
    }
}

fn crossed_params() -> Params {
    let mut params = params(4);
    for (i, (x, y)) in [(0., 0.), (10., 0.), (0., 11.), (11., 0.), (0., 10.)]
        .into_iter()
        .enumerate()
    {
        params.cli[i].coord_x = x;
        params.cli[i].coord_y = y;
        for (j, (a, b)) in [(0., 0.), (10., 0.), (0., 11.), (11., 0.), (0., 10.)]
            .into_iter()
            .enumerate()
        {
            params.time_cost.set(i, j, (x - a).hypot(y - b));
        }
    }
    params.config.swap_star_demand_percentage = 0;
    params.config.swap_star_full_pass_interval = 2;
    params
}

#[test]
fn broad_pass_recovers_an_improving_exchange_omitted_by_filtering() {
    let params = crossed_params();
    let mut individual = Individual::from_string(&params, "1 2 0 3 4");
    let before = individual.my_cost_sol.penalized_cost;
    let mut engine = HGSLocalSearch::new(&params);
    engine.penalty_capacity_ls = params.penalty_capacity.get();
    engine.penalty_time_warp_ls = params.penalty_time_warp.get();
    engine.load_individual(&individual);
    engine.swap_neighborhood.begin_pass(); // broad
    engine.swap_neighborhood.begin_pass(); // filtered, empty shortlist
    assert!(!engine.swap_star(0, 1, false));
    assert_eq!(engine.swap_pairs_tested.get(), 0);
    assert!(!engine.best_insert_initialized_for_route[0]);
    assert!(!engine.best_insert_initialized_for_route[1]);
    assert!(engine.swap_neighborhood.begin_pass()); // force broad refresh
    assert!(engine.swap_star(0, 1, false));
    assert!(engine.swap_pairs_tested.get() > 0);
    engine.export_individual(&mut individual);
    assert!(individual.my_cost_sol.penalized_cost < before - 1.);
    assert!(individual.is_feasible);
    let mut covered: Vec<_> = individual.chrom_r.iter().flatten().copied().collect();
    covered.sort_unstable();
    assert_eq!(covered, vec![1, 2, 3, 4]);
}

#[test]
fn full_shortlists_match_broad_swap_results_including_time_window_evaluation() {
    let mut params = crossed_params();
    params.config.swap_star_neighbors = 4;
    params.config.swap_star_demand_percentage = 100;
    let initial = Individual::from_string(&params, "1 2 0 3 4");
    for with_tw in [false, true] {
        let mut outcomes = Vec::new();
        for filtered in [false, true] {
            let mut individual = initial.clone();
            let mut engine = HGSLocalSearch::new(&params);
            engine.penalty_capacity_ls = params.penalty_capacity.get();
            engine.penalty_time_warp_ls = params.penalty_time_warp.get();
            engine.load_individual(&individual);
            engine.swap_neighborhood.begin_pass();
            if filtered {
                engine.swap_neighborhood.begin_pass();
            }
            assert!(engine.swap_star(0, 1, with_tw));
            engine.export_individual(&mut individual);
            outcomes.push((individual.chrom_r, individual.my_cost_sol.penalized_cost));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn filtered_candidates_follow_current_route_membership_after_moves() {
    let mut params = params(4);
    params.config.swap_star_neighbors = 1;
    params.config.swap_star_demand_percentage = 100;
    // Only customer 3 is selected for customer 1, in either direction.
    for i in 1..=4 {
        for j in 1..=4 {
            params.time_cost.set(i, j, 100.);
        }
    }
    for (i, j) in [(1, 3), (3, 1), (2, 4), (4, 2)] {
        params.time_cost.set(i, j, 1.);
    }
    let individual = Individual::from_string(&params, "1 2 0 3 4");
    let mut engine = HGSLocalSearch::new(&params);
    engine.load_individual(&individual);
    engine.swap_neighborhood.begin_pass();
    engine.swap_neighborhood.begin_pass();
    let first = engine.nodes[engine.depot_start(1)].next as usize;
    assert_eq!(
        engine
            .swap_neighborhood
            .candidates(1, first, 1, &engine.nodes)
            .collect::<Vec<_>>(),
        vec![3]
    );
    engine.insert_node(3, 1);
    assert_eq!(
        engine
            .swap_neighborhood
            .candidates(1, first, 1, &engine.nodes)
            .count(),
        0
    );
}

#[test]
fn default_demand_filter_reduces_swap_pair_evaluations_in_both_cost_modes() {
    let mut params = params(40);
    params.config.swap_star_neighbors = 35;
    params.config.swap_star_demand_percentage = 30;
    let mut individual = Individual::new_empty(&params);
    individual.chrom_r[0] = (1..=20).collect();
    individual.chrom_r[1] = (21..=40).collect();
    individual.evaluate_complete_cost();
    for with_tw in [false, true] {
        let mut counts = Vec::new();
        for filtered in [false, true] {
            let mut engine = HGSLocalSearch::new(&params);
            engine.penalty_capacity_ls = params.penalty_capacity.get();
            engine.penalty_time_warp_ls = params.penalty_time_warp.get();
            engine.load_individual(&individual);
            engine.swap_neighborhood.begin_pass();
            if filtered {
                engine.swap_neighborhood.begin_pass();
            }
            assert!(!engine.swap_star(0, 1, with_tw));
            counts.push(engine.swap_pairs_tested.get());
        }
        assert_eq!(counts[0], 400);
        assert!(
            counts[1] > 0 && counts[1] < counts[0],
            "pair counts: {counts:?}"
        );
    }
}
