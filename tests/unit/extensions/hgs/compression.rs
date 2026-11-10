use super::model::Plan;
use super::*;
use crate::extensions::hgs::Individual;
use crate::inputs::{Customer, Rounding};

fn instance(n: usize) -> Instance {
    Instance {
        name: "compression".into(),
        vehicles: 3,
        capacity: 3.,
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
    }
}
fn individual<'p>(params: &'p Params, routes: &[Vec<usize>]) -> Individual<'p> {
    let mut i = Individual::new_empty(params);
    for (dst, src) in i.chrom_r.iter_mut().zip(routes) {
        dst.clone_from(src);
    }
    i.chrom_t = routes.iter().flatten().copied().collect();
    i.evaluate_complete_cost();
    i
}
fn compare(root: &Params, compact: &Params, plan: &Plan, routes: &[Vec<usize>]) {
    let expansion = Expansion::new(root.nb_clients as usize).contract(plan);
    let a = individual(root, &expansion.expand(routes));
    let b = individual(compact, routes);
    assert!((a.my_cost_sol.distance - b.my_cost_sol.distance - plan.frozen_distance).abs() < 1e-7);
    assert!(
        (a.my_cost_sol.time_warp - b.my_cost_sol.time_warp).abs() < 1e-7,
        "warps {} != {}, routes={routes:?}",
        a.my_cost_sol.time_warp,
        b.my_cost_sol.time_warp
    );
    assert_eq!(a.my_cost_sol.capacity_excess, b.my_cost_sol.capacity_excess);
    assert_eq!(a.is_feasible, b.is_feasible);
}
fn permutations(v: &mut [usize], start: usize, visit: &mut impl FnMut(&[usize])) {
    if start == v.len() {
        visit(v);
        return;
    }
    for i in start..v.len() {
        v.swap(i, start);
        permutations(v, start + 1, visit);
        v.swap(i, start);
    }
}

#[test]
fn fractional_asymmetric_macro_nodes_match_forward_replay_for_every_partition() {
    let mut state = 37_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    for _ in 0..100 {
        let mut root = Params::from_instance(&instance(6));
        root.vehicle_capacity = 3 + (next() % 10) as i32;
        for i in 0..=6 {
            for j in 0..=6 {
                root.time_cost.set(
                    i,
                    j,
                    if i == j {
                        0.
                    } else {
                        (next() % 1000) as f64 / 100. + 0.125
                    },
                );
            }
        }
        for i in 1..=6 {
            root.cli[i].earliest_arrival = (next() % 3000) as f64 / 100.;
            root.cli[i].latest_arrival = root.cli[i].earliest_arrival + 50.125;
            root.cli[i].service_duration = (next() % 500) as f64 / 100.;
            root.cli[i].release_time = (next() % 15000) as f64 / 100.;
            root.cli[i].demand = 1 + (next() % 4) as i32;
        }
        root.cli[0].latest_arrival = 100.75;
        let plan = Plan::build(
            &root,
            vec![vec![1, 2], vec![3, 4], vec![5], vec![6]],
            vec![],
        )
        .unwrap();
        let compact = plan.parameters(&root).unwrap();
        assert_eq!(compact.nb_clients, 4);
        permutations(&mut [1, 2, 3, 4], 0, &mut |tour| {
            for mask in 0_u32..8 {
                if mask.count_ones() > 2 {
                    continue;
                }
                let mut routes = vec![vec![]];
                for (i, &v) in tour.iter().enumerate() {
                    routes.last_mut().unwrap().push(v);
                    if mask & (1 << i) != 0 {
                        routes.push(vec![]);
                    }
                }
                compare(&root, &compact, &plan, &routes);
            }
        });
    }
}

#[test]
fn nested_compression_and_frozen_routes_do_not_double_count_distance() {
    let root = Params::from_instance(&instance(6));
    let first = Plan::build(
        &root,
        vec![vec![1, 2], vec![3], vec![4], vec![5], vec![6]],
        vec![],
    )
    .unwrap();
    let middle = first.parameters(&root).unwrap();
    let second = Plan::build(&middle, vec![vec![1, 2], vec![3]], vec![vec![4, 5]]).unwrap();
    let compact = second.parameters(&middle).unwrap();
    let map = Expansion::new(6).contract(&first).contract(&second);
    let routes = vec![vec![1, 2]];
    let expanded = map.expand(&routes);
    assert_eq!(expanded, vec![vec![1, 2, 3, 4], vec![5, 6]]);
    let a = individual(&root, &expanded);
    let b = individual(&compact, &routes);
    assert!(
        (a.my_cost_sol.distance - b.my_cost_sol.distance - second.frozen_distance).abs() < 1e-9
    );
    assert_eq!(compact.nb_vehicles, 2);
    assert_eq!(
        second.project(&first.project(&expanded).unwrap()).unwrap(),
        routes
    );
}

#[test]
fn nonrepresentable_chains_stay_singletons_and_partial_chains_are_rejected() {
    let mut root = Params::from_instance(&instance(4));
    root.cli[1].earliest_arrival = 200.;
    root.cli[1].latest_arrival = 201.;
    root.cli[2].earliest_arrival = 0.;
    root.cli[2].latest_arrival = 1.;
    let plan = Plan::build(&root, vec![vec![1, 2], vec![3, 4]], vec![]).unwrap();
    assert_eq!(plan.chains, vec![vec![1], vec![2], vec![3, 4]]);
    assert!(plan.project(&[vec![1, 3], vec![4, 2]]).is_none());
    root.cli[1].earliest_arrival = 0.;
    root.cli[1].latest_arrival = 1000.;
    root.cli[1].n_tw = 2;
    root.cli[1].time_windows = vec![
        crate::extensions::hgs::TimeWindow::new(0., 100.),
        crate::extensions::hgs::TimeWindow::new(300., 1000.),
    ];
    let plan = Plan::build(&root, vec![vec![1, 2], vec![3, 4]], vec![]).unwrap();
    let compact = plan.parameters(&root).unwrap();
    assert_eq!(compact.cli[1].n_tw, 2);
    compare(&root, &compact, &plan, &[vec![1, 2, 3]]);
    root.penalty_wait_time = 1.;
    assert!(Plan::build(&root, vec![vec![1, 2], vec![3, 4]], vec![]).is_none());
}

#[test]
fn consensus_remembers_feasible_observations_even_after_they_leave_the_population() {
    let root = Params::from_instance(&instance(4));
    let a = individual(&root, &[vec![1, 2], vec![3, 4]]);
    let b = individual(&root, &[vec![1, 3], vec![2, 4]]);
    let mut population = Population::new(&root);
    population.add_individual(&a, true);
    population.consensus = Some(Consensus::new(4));
    population.add_individual(&a, true);
    population.add_individual(&b, true);
    population.feasible.retain(|i| i.successors == a.successors);
    assert!(population.compression_plan().is_none());
}

#[test]
fn infeasible_survivors_block_incompatible_chain_fixing() {
    let mut root = Params::from_instance(&instance(4));
    root.vehicle_capacity = 2;
    let a = individual(&root, &[vec![1, 2], vec![3, 4]]);
    let b = individual(&root, &[vec![1, 3, 2, 4]]);
    assert!(!b.is_feasible);
    let mut population = Population::new(&root);
    population.add_individual(&a, true);
    population.consensus = Some(Consensus::new(4));
    population.add_individual(&a, true);
    population.add_individual(&b, true);
    assert!(population.compression_plan().is_none());
}

#[test]
fn migration_preserves_population_incumbent_penalties_history_rng_and_clock() {
    let root = Params::from_instance(&instance(6));
    let a = individual(&root, &[vec![1, 2, 3], vec![4, 5, 6]]);
    let b = individual(&root, &[vec![1, 2, 4], vec![3, 5, 6]]);
    let infeasible = individual(&root, &[vec![1, 2, 3, 4, 5, 6]]);
    let mut population = Population::new(&root);
    population.minimum_population_size = 33;
    population.add_individual(&a, true);
    population.consensus = Some(Consensus::new(6));
    population.add_individual(&a, true);
    population.add_individual(&b, true);
    for _ in 0..90 {
        population.add_individual(&infeasible, true);
    }
    population.nb_generations = 140;
    let plan = population.compression_plan().unwrap();
    let age = root.elapsed_secs();
    let compact = plan.parameters(&root).unwrap();
    assert!(compact.elapsed_secs() >= age);
    assert_eq!(compact.nb_clients, 4);
    let rng = root.rng.borrow().clone().next();
    assert_eq!(compact.rng.borrow_mut().next(), rng);
    let old_size = (population.feasible.len(), population.infeasible.len());
    let mut restored = population
        .checkpoint()
        .project(&compact, &plan)
        .unwrap()
        .restore(&compact);
    assert_eq!(
        (restored.feasible.len(), restored.infeasible.len()),
        old_size
    );
    assert_eq!(restored.nb_generations, 140);
    assert_eq!(restored.minimum_population_size, 33);
    let old_best = population.get_best_found().unwrap();
    let new_best = restored.get_best_found().unwrap();
    assert!((old_best.my_cost_sol.distance - new_best.my_cost_sol.distance).abs() < 1e-8);
    population.manage_penalties();
    restored.manage_penalties();
    assert_eq!(root.penalty_capacity.get(), compact.penalty_capacity.get());
    assert_eq!(
        root.penalty_time_warp.get(),
        compact.penalty_time_warp.get()
    );
}

#[test]
fn epoch_boundaries_preserve_the_uncompressed_search_trajectory() {
    let mut outcomes = Vec::new();
    for bounded in [false, true] {
        let mut params = Params::from_instance(&instance(6));
        params.verbose = false;
        params.set_seed(9);
        let mut split = Split::new(&params);
        let mut local = HGSLocalSearch::new(&params);
        let mut population = Population::new(&params);
        population.generate_population(&mut split, &mut local);
        let mut genetic = Genetic::new(&params, &mut split, &mut local);
        if bounded {
            let mut state = EvolutionState::default();
            while !genetic.run_epoch(&mut population, 30, &mut state, 7) {}
        } else {
            genetic.run(&mut population, 30);
        }
        outcomes.push((
            population.get_best_found().unwrap().chrom_r.clone(),
            population.nb_generations,
            params.rng.borrow_mut().next(),
        ));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}

#[test]
fn compressed_api_preserves_replayed_witnesses_and_has_deterministic_fixed_work() {
    let input = instance(6);
    let options = HgsOptions {
        time_limit: 0.,
        nb_iter: 100,
        seed: 7,
        ..Default::default()
    };
    let a = run_hgs_compressed(&input, &options, &CompressionOptions { interval: 5 }).unwrap();
    let b = run_hgs_compressed(&input, &options, &CompressionOptions { interval: 5 }).unwrap();
    assert_eq!(a.solution.routes, b.solution.routes);
    assert_eq!(a.solution.generations, b.solution.generations);
    assert_eq!(a.solution.best_cost, b.solution.best_cost);
    assert!(a.compression.checks > 0);
    assert_eq!(
        super::super::solution::validate_plan(&input, &a.solution.routes).unwrap(),
        a.solution.best_cost.unwrap()
    );
    let no_checks = run_hgs_compressed(
        &input,
        &options,
        &CompressionOptions {
            interval: usize::MAX,
        },
    )
    .unwrap();
    let original = super::super::driver::run_hgs(&input, &options).unwrap();
    assert_eq!(no_checks.solution.routes, original.routes);
    assert_eq!(no_checks.solution.generations, original.generations);
    assert_eq!(no_checks.compression.contractions, 0);
}

#[test]
fn complete_consensus_routes_form_a_terminal_compression_without_losing_customers() {
    let root = Params::from_instance(&instance(4));
    let a = individual(&root, &[vec![1, 2], vec![3, 4]]);
    let mut population = Population::new(&root);
    population.add_individual(&a, true);
    population.consensus = Some(Consensus::new(4));
    population.add_individual(&a, true);
    let plan = population.compression_plan().unwrap();
    assert!(plan.chains.is_empty());
    let map = Expansion::new(4).contract(&plan);
    let expanded = map.expand(&[]);
    assert_eq!(expanded, vec![vec![1, 2], vec![3, 4]]);
    assert_eq!(plan.project(&expanded).unwrap(), Vec::<Vec<usize>>::new());
    assert_eq!(plan.frozen_distance, a.my_cost_sol.distance);
}

#[test]
fn invalid_compression_options_are_rejected_before_search() {
    let input = instance(4);
    assert!(run_hgs_compressed(
        &input,
        &HgsOptions::default(),
        &CompressionOptions { interval: 0 }
    )
    .is_err());
    for limit in [-1., f64::NAN, f64::INFINITY] {
        assert!(run_hgs_compressed(
            &input,
            &HgsOptions {
                time_limit: limit,
                ..Default::default()
            },
            &CompressionOptions::default()
        )
        .is_err());
    }
    let mut invalid = input;
    invalid.capacity = 0.;
    assert!(run_hgs_compressed(
        &invalid,
        &HgsOptions::default(),
        &CompressionOptions::default()
    )
    .is_err());
}

#[test]
fn progress_reports_full_objective_after_freezing_routes() {
    let root = Params::from_instance(&instance(6));
    let a = individual(&root, &[vec![1, 2], vec![3, 4], vec![5, 6]]);
    let b = individual(&root, &[vec![1, 2], vec![3, 5], vec![4, 6]]);
    let mut population = Population::new(&root);
    population.add_individual(&a, true);
    population.consensus = Some(Consensus::new(6));
    population.add_individual(&a, true);
    population.add_individual(&b, true);
    let plan = population.compression_plan().unwrap();
    assert_eq!(plan.fixed, vec![vec![1, 2]]);
    let compact = plan.parameters(&root).unwrap();
    let mut restored = population
        .checkpoint()
        .project(&compact, &plan)
        .unwrap()
        .restore(&compact);
    let improved = individual(&compact, &[vec![1], vec![2, 3, 4]]);
    assert!(restored.add_individual(&improved, true));
    let original_routes = Expansion::new(6).contract(&plan).expand(&improved.chrom_r);
    let expected = individual(&root, &original_routes)
        .my_cost_sol
        .penalized_cost;
    assert!((restored.search_progress.last().unwrap().1 - expected).abs() < 1e-9);
}

#[test]
fn freezing_never_discards_a_small_positive_time_warp_penalty() {
    let mut root = Params::from_instance(&instance(1));
    root.cli[1].latest_arrival = root.time_cost.get(0, 1) - 1e-6;
    let routed = individual(&root, &[vec![1]]);
    assert!(routed.is_feasible); // within the HGS feasibility comparison band
    assert!(routed.my_cost_sol.time_warp > 0.);
    assert!(Plan::build(&root, vec![], vec![vec![1]]).is_none());
}

#[test]
fn reserved_seed_is_an_incumbent_without_entering_the_pool_and_is_inserted_once() {
    let root = Params::from_instance(&instance(6));
    let seed = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let mut local = HGSLocalSearch::new(&root);
    let mut population = Population::new(&root);
    population.initialize_seeds(std::slice::from_ref(&seed), &mut local, true);
    assert!(population.feasible.is_empty() && population.infeasible.is_empty());
    assert_eq!(population.get_best_found().unwrap().chrom_t, seed.concat());
    assert_eq!(population.insert_reserved_seeds(1, 20), None);
    let worse = individual(&root, &[vec![1, 3, 5], vec![2, 4, 6]]);
    population.add_individual(&worse, true);
    assert_eq!(population.insert_reserved_seeds(9, 20), None);
    assert_eq!(population.insert_reserved_seeds(10, 20), Some(true));
    assert!(population.reserved_seeds.is_empty());
    assert_eq!(
        population.get_best_feasible().unwrap().chrom_t,
        seed.concat()
    );
    let count = population.feasible.len();
    assert_eq!(population.insert_reserved_seeds(20, 20), None);
    assert_eq!(population.feasible.len(), count);
}

#[test]
fn reaching_seed_cost_or_half_time_inserts_without_waiting_for_stagnation() {
    for caught_up in [false, true] {
        let mut root = Params::from_instance(&instance(4));
        if !caught_up {
            // Deterministically past half the budget, without sleeping.
            root.config.time_limit = f64::MIN_POSITIVE;
        }
        let seed = vec![vec![1, 2], vec![3, 4]];
        let mut local = HGSLocalSearch::new(&root);
        let mut population = Population::new(&root);
        population.initialize_seeds(std::slice::from_ref(&seed), &mut local, true);
        if caught_up {
            population.add_individual(&individual(&root, &seed), true);
        }
        assert!(population.insert_reserved_seeds(1, 100).is_some());
        assert!(population.reserved_seeds.is_empty());
    }
}

#[test]
fn every_reserved_seed_blocks_incompatible_compression_even_if_not_the_incumbent() {
    let root = Params::from_instance(&instance(4));
    let a = individual(&root, &[vec![1, 2], vec![3, 4]]);
    let b = individual(&root, &[vec![1, 3], vec![2, 4]]);
    let mut population = Population::new(&root);
    population.add_individual(&a, true);
    population.reserved_seeds.push(b);
    population.consensus = Some(Consensus::new(4));
    population.add_individual(&a, true);
    assert!(population.compression_plan().is_none());
    assert_eq!(population.reserved_seeds.len(), 1);
}

#[test]
fn pending_seed_survives_projection_with_frozen_routes_and_later_insertion() {
    let root = Params::from_instance(&instance(6));
    let seed = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let mut population = Population::new(&root);
    population.initialize_seeds(
        std::slice::from_ref(&seed),
        &mut HGSLocalSearch::new(&root),
        true,
    );
    let breeding = individual(&root, &[vec![1, 2], vec![3, 5], vec![4, 6]]);
    population.add_individual(&breeding, true);
    population.consensus = Some(Consensus::new(6));
    population.add_individual(&breeding, true);
    let plan = population.compression_plan().unwrap();
    assert_eq!(plan.fixed, vec![vec![1, 2]]);
    let compact = plan.parameters(&root).unwrap();
    let mut restored = population
        .checkpoint()
        .project(&compact, &plan)
        .unwrap()
        .restore(&compact);
    assert_eq!(restored.reserved_seeds.len(), 1);
    let expanded = Expansion::new(6)
        .contract(&plan)
        .expand(&restored.reserved_seeds[0].chrom_r);
    let mut expected = seed.clone();
    let mut actual = expanded;
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected);
    assert_eq!(restored.insert_reserved_seeds(10, 20), Some(true));
    assert!(restored.reserved_seeds.is_empty());
    assert!(restored.get_best_feasible().unwrap().is_feasible);
}

#[test]
fn tiny_deadline_preserves_the_exact_seed_with_and_without_compression() {
    let input = instance(6);
    let seed = vec![vec![1, 3, 5], vec![2, 4, 6]];
    for compressed in [false, true] {
        let options = HgsOptions {
            time_limit: f64::MIN_POSITIVE,
            delay_seed: true,
            compression: compressed.then(|| CompressionOptions { interval: 1 }),
            ..Default::default()
        };
        let result =
            super::super::run_hgs_seeded(&input, &options, std::slice::from_ref(&seed)).unwrap();
        assert_eq!(result.routes, seed);
        assert_eq!(result.generations, 0);
        assert_eq!(
            result.best_cost,
            Some(super::super::solution::validate_plan(&input, &seed).unwrap())
        );
    }
}

#[test]
fn seeded_compression_is_deterministic_and_returns_original_customer_ids() {
    let input = instance(6);
    let seed = vec![vec![1, 3, 5], vec![2, 4, 6]];
    let options = HgsOptions {
        time_limit: 0.,
        nb_iter: 60,
        population_size: 3,
        seed: 7,
        delay_seed: true,
        compression: Some(CompressionOptions { interval: 2 }),
        ..Default::default()
    };
    let a = super::super::run_hgs_seeded(&input, &options, std::slice::from_ref(&seed)).unwrap();
    let b = super::super::run_hgs_seeded(&input, &options, std::slice::from_ref(&seed)).unwrap();
    assert_eq!(a.routes, b.routes);
    assert_eq!(a.generations, b.generations);
    let cost = super::super::solution::validate_plan(&input, &a.routes).unwrap();
    assert_eq!(a.best_cost, Some(cost));
    assert!(cost <= super::super::solution::validate_plan(&input, &seed).unwrap());
    let collected = run_seeded(
        &input,
        &options,
        options.compression.as_ref().unwrap(),
        std::slice::from_ref(&seed),
        true,
    )
    .unwrap();
    assert!(collected.compression.checks > 0);
    for route in &collected.solution.columns {
        let mut unique = route.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), route.len());
        assert!(route.iter().all(|&c| (1..=6).contains(&c)));
    }
    assert!(seed.iter().all(|r| collected.solution.columns.contains(r)));
}

#[test]
fn compressed_seed_validation_rejects_missing_duplicate_and_out_of_range_customers() {
    let input = instance(4);
    let options = HgsOptions {
        compression: Some(CompressionOptions { interval: 1 }),
        delay_seed: true,
        ..Default::default()
    };
    for seed in [
        vec![vec![1, 2], vec![3]],
        vec![vec![1, 2], vec![2, 3, 4]],
        vec![vec![1, 2], vec![3, 5]],
    ] {
        assert!(super::super::run_hgs_seeded(&input, &options, &[seed]).is_err());
    }
}

#[test]
fn collected_routes_expand_a_terminal_seed_compression() {
    let mut input = instance(2);
    for c in 1..=2 {
        input.customers[c].ready = c as f64;
        input.customers[c].due = c as f64;
    }
    let seed = vec![vec![1, 2]];
    let options = HgsOptions {
        time_limit: 0.,
        nb_iter: 60,
        population_size: 3,
        seed: 7,
        delay_seed: true,
        ..Default::default()
    };
    let found = run_seeded(
        &input,
        &options,
        &CompressionOptions { interval: 2 },
        std::slice::from_ref(&seed),
        true,
    )
    .unwrap();
    assert_eq!(found.compression.final_customers, 0);
    assert_eq!(found.compression.frozen_routes, 1);
    assert_eq!(found.compression.contractions, 1);
    assert_eq!(found.solution.routes, seed);
    assert_eq!(found.solution.columns, seed);
    assert_eq!(found.solution.best_cost, Some(4.));
}

#[test]
fn total_generation_caps_survive_compression_epochs_and_seed_reservation() {
    let data = instance(6);
    let seed = vec![vec![1, 3, 5], vec![2, 4, 6]];
    for cap in [0, 1, 3, 10] {
        let options = HgsOptions {
            time_limit: 0.,
            nb_iter: 1000,
            population_size: 3,
            generation_limit: Some(cap),
            delay_seed: true,
            compression: Some(CompressionOptions { interval: 1 }),
            ..Default::default()
        };
        let result =
            super::super::run_hgs_seeded(&data, &options, std::slice::from_ref(&seed)).unwrap();
        assert!(result.generations <= cap);
        let cost = super::super::solution::validate_plan(&data, &result.routes).unwrap();
        assert_eq!(result.best_cost, Some(cost));
        assert!(cost <= super::super::solution::validate_plan(&data, &seed).unwrap());
    }
}
