use super::*;

fn problem() -> Problem {
    Problem {
        demands: vec![1.0, 2.0, 1.0],
        capacity: 4.0,
        costs: vec![vec![Some(1.0); 5]; 5],
    }
}

fn options(p: usize) -> EnumerationOptions {
    EnumerationOptions {
        p,
        ..Default::default()
    }
}

#[test]
fn compact_extremes_and_coefficients() {
    let f = problem().enumerate(options(2)).unwrap();
    let columns: Vec<_> = f
        .steps()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.path == [1, 2, 3])
        .collect();
    // Load exactly Q: duplicate zero/max extreme removed.
    assert_eq!(columns.len(), 1);
    assert_eq!(
        f.entries(columns[0].0).unwrap(),
        vec![
            (0, 1.0),
            (1, 2.0),
            (2, 1.0),
            (3, 1.0),
            (5, -1.0),
            (6, 1.0),
            (8, -4.0)
        ]
    );
    let ends: Vec<_> = f.steps().iter().filter(|s| s.path == [1, 3, 4]).collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].prior_demand, 2.0);
    assert!(f.steps().iter().any(|s| s.path == [0, 1]));
    assert!(!f.steps().iter().any(|s| s.path == [1, 4]));
    assert!(!f.steps().iter().any(|s| s.path == [0, 4]));
    let internal: Vec<_> = f.steps().iter().filter(|s| s.path == [1, 3, 2]).collect();
    assert_eq!(internal.len(), 1);
    let mut p = problem();
    p.capacity = 5.0;
    let f = p.enumerate(options(2)).unwrap();
    let priors: Vec<_> = f
        .steps()
        .iter()
        .filter(|s| s.path == [1, 2, 3])
        .map(|s| s.prior_demand)
        .collect();
    assert_eq!(priors, vec![0.0, 1.0]);
}

#[test]
fn arc_and_route_extremes() {
    let p = problem();
    let arcs = p.enumerate(options(1)).unwrap();
    assert!(arcs.steps().iter().all(|s| s.path.len() == 2));
    let routes = p.enumerate(options(4)).unwrap();
    assert_eq!(routes.steps().len(), 30); // 15 routes and 15 depot prefixes
    assert!(routes.steps().iter().all(|s| s.path[0] == 0));
    assert_eq!(
        routes
            .steps()
            .iter()
            .filter(|s| s.path.last() == Some(&4))
            .count(),
        15
    );
    let large = p.enumerate(options(usize::MAX)).unwrap();
    assert_eq!(routes.steps().len(), large.steps().len());
}

#[test]
fn invalid_inputs_and_work_limits() {
    assert!(problem().enumerate(options(0)).is_err());
    assert!(problem()
        .enumerate(EnumerationOptions {
            max_columns: 1,
            ..options(2)
        })
        .is_err());
    assert!(problem()
        .enumerate(EnumerationOptions {
            max_extensions: 1,
            ..options(2)
        })
        .is_err());
    let mut p = problem();
    p.demands[0] = 0.0;
    assert!(p.enumerate(options(2)).is_err());
    p.demands[0] = f64::NAN;
    assert!(p.validate().is_err());
    p = problem();
    p.costs.pop();
    assert!(p.validate().is_err());
}

#[test]
fn sparse_directed_paths_and_capacity() {
    let mut p = problem();
    p.capacity = 2.0;
    p.costs = vec![vec![None; 5]; 5];
    for (i, j) in [(0, 1), (1, 3), (3, 4)] {
        p.costs[i][j] = Some(1.0);
    }
    let f = p.enumerate(options(2)).unwrap();
    assert!(f.steps().iter().any(|s| s.path == [1, 3, 4]));
    assert!(f.steps().iter().all(|s| s.demand <= 2.0));
    assert!(f
        .steps()
        .iter()
        .all(|s| s.path.windows(2).all(|a| p.costs[a[0]][a[1]].is_some())));
}

#[test]
fn bucket_pricing_matches_every_enumerated_job_and_parallel_order() {
    let mut problem = problem();
    // Deterministic varying duals of both signs expose all load-cost terms.
    for seed in 0..5 {
        problem.capacity = if seed % 2 == 0 { 4.0 } else { 3.0 };
        let duals: Vec<_> = (0..9)
            .map(|i| ((i * 7 + seed * 11) % 17) as f64 - 8.0)
            .collect();
        for p in 1..=5 {
            for weight in [0.0, 1.0] {
                let full = problem.enumerate(options(p)).unwrap();
                let minimum = full
                    .steps()
                    .iter()
                    .enumerate()
                    .map(|(c, s)| {
                        weight * s.cost
                            - full
                                .entries(c)
                                .unwrap()
                                .iter()
                                .map(|&(r, a)| a * duals[r])
                                .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                let settings = PricingOptions {
                    p,
                    threads: 1,
                    ..Default::default()
                };
                let serial = problem.price_steps(&duals, weight, &settings).unwrap();
                let parallel = problem
                    .price_steps(
                        &duals,
                        weight,
                        &PricingOptions {
                            threads: 3,
                            ..settings
                        },
                    )
                    .unwrap();
                assert!(
                    (serial.minimum_reduced_cost.unwrap() - minimum).abs() < 1e-7,
                    "seed={seed} p={p} weight={weight} actual={:?} expected={minimum}",
                    serial.minimum_reduced_cost
                );
                assert_eq!(serial.minimum_reduced_cost, parallel.minimum_reduced_cost);
                assert_eq!(serial.generated_labels, parallel.generated_labels);
                assert_eq!(
                    serial
                        .steps
                        .iter()
                        .map(|s| (&s.path, s.prior_demand))
                        .collect::<Vec<_>>(),
                    parallel
                        .steps
                        .iter()
                        .map(|s| (&s.path, s.prior_demand))
                        .collect::<Vec<_>>()
                );
                for step in &serial.steps {
                    let candidates = full.steps().iter().enumerate().filter(|(_, s)| {
                        s.path.first() == step.path.first()
                            && s.path.last() == step.path.last()
                            && (s.prior_demand == 0.0) == (step.prior_demand == 0.0)
                    });
                    let actual = weight * step.cost
                        - step_entries(&problem.demands, step)
                            .iter()
                            .map(|&(r, a)| a * duals[r])
                            .sum::<f64>();
                    for (c, s) in candidates {
                        let expected = weight * s.cost
                            - full
                                .entries(c)
                                .unwrap()
                                .iter()
                                .map(|&(r, a)| a * duals[r])
                                .sum::<f64>();
                        // Coincident extremes can also belong to the other job.
                        if step.prior_demand == 0.0 && step.demand == problem.capacity {
                            continue;
                        }
                        assert!(actual <= expected + 1e-7);
                    }
                }
            }
        }
    }
    let settings = PricingOptions {
        p: 2,
        threads: 2,
        max_labels_per_job: 1,
        ..Default::default()
    };
    assert!(problem.price_steps(&[0.0; 9], 1.0, &settings).is_err());
    assert!(problem
        .price_steps(
            &[0.0; 9],
            1.0,
            &PricingOptions {
                max_labels_per_job: 1000,
                time_limit: std::time::Duration::ZERO,
                ..settings
            }
        )
        .is_err());
}

#[test]
fn long_steps_exclude_profitable_cycles_exactly() {
    let problem = Problem {
        demands: vec![1.0; 5],
        capacity: 6.0,
        costs: vec![vec![Some(1.0); 7]; 7],
    };
    let mut duals = vec![0.0; 15];
    duals[1] = 100.0;
    duals[2] = 100.0;
    for p in [4, 5] {
        let full = problem.enumerate(options(p)).unwrap();
        let expected = full
            .steps()
            .iter()
            .enumerate()
            .map(|(c, s)| {
                s.cost
                    - full
                        .entries(c)
                        .unwrap()
                        .iter()
                        .map(|&(r, a)| a * duals[r])
                        .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        let result = problem
            .price_steps(
                &duals,
                1.0,
                &PricingOptions {
                    p,
                    threads: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!((result.minimum_reduced_cost.unwrap() - expected).abs() < 1e-7);
        for step in result.steps {
            let unique: std::collections::BTreeSet<_> = step.path.iter().collect();
            assert_eq!(unique.len(), step.path.len());
        }
        assert!(problem
            .price_steps(
                &duals,
                1.0,
                &PricingOptions {
                    p,
                    threads: 2,
                    max_subproblems_per_job: 1,
                    ..Default::default()
                }
            )
            .is_err());
    }
}

#[cfg(feature = "clp")]
#[test]
fn parallel_column_generation_matches_full_pool() {
    use crate::master::clp::ClpMaster;
    let mut problem = problem();
    for i in 0..5 {
        for j in 0..5 {
            problem.costs[i][j] = Some(((i * 7 + j * 3 + i * j) % 11 + 1) as f64);
        }
    }
    for p in 1..=4 {
        let mut full = problem
            .enumerate(options(p))
            .unwrap()
            .into_master(ClpMaster::new().unwrap())
            .unwrap();
        assert!(full.solve());
        for threads in [1, 3] {
            let result = problem
                .solve_column_generation(
                    ClpMaster::new().unwrap(),
                    ColumnGenerationOptions {
                        pricing: PricingOptions {
                            p,
                            threads,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(result.status, ColumnGenerationStatus::Converged);
            assert!(
                (result.lp_bound.unwrap() - full.objective()).abs() < 1e-6,
                "p={p} threads={threads} {result:?}"
            );
        }
    }
    let result = problem
        .solve_column_generation(
            ClpMaster::new().unwrap(),
            ColumnGenerationOptions {
                max_iterations: 0,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.status, ColumnGenerationStatus::IterationLimit);
    assert!(result.lp_bound.is_none());
    let result = problem
        .solve_column_generation(
            ClpMaster::new().unwrap(),
            ColumnGenerationOptions {
                time_limit: std::time::Duration::ZERO,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.status, ColumnGenerationStatus::TimeLimit);
    assert!(result.lp_bound.is_none());
    for row in &mut problem.costs {
        row[4] = None;
    }
    let result = problem
        .solve_column_generation(
            ClpMaster::new().unwrap(),
            ColumnGenerationOptions::default(),
        )
        .unwrap();
    assert_eq!(result.status, ColumnGenerationStatus::Infeasible);
    assert!(result.lp_bound.is_none());
}

#[cfg(feature = "clp")]
#[test]
fn cvrplib_parallel_pricing_matches_full_pstep_lp() {
    use crate::master::clp::ClpMaster;
    for text in [
        include_str!("../../../instances/cvrplib/P-n16-k8.vrp"),
        include_str!("../../../instances/cvrplib/E-n22-k4.vrp"),
    ] {
        let instance = crate::inputs::Instance::parse(text).unwrap();
        let problem = Problem::from_instance(&instance).unwrap();
        let mut full = problem
            .enumerate(options(2))
            .unwrap()
            .into_master(ClpMaster::new().unwrap())
            .unwrap();
        assert!(full.solve());
        let result = problem
            .solve_column_generation(
                ClpMaster::new().unwrap(),
                ColumnGenerationOptions {
                    pricing: PricingOptions {
                        p: 2,
                        threads: 4,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(result.status, ColumnGenerationStatus::Converged);
        assert_eq!(result.pricing_workers, 4);
        assert!((result.lp_bound.unwrap() - full.objective()).abs() < 1e-6);
    }
}

#[cfg(feature = "clp")]
#[test]
fn lp_bounds_against_independent_set_partitioning() {
    use crate::master::clp::ClpMaster;
    // Three unit customers with Q=2: three half-weight two-customer routes
    // cost 4.5. Integer optimum costs 5 (two routes).
    let mut p = problem();
    p.demands = vec![1.0; 3];
    p.capacity = 2.0;
    for step_size in 1..=4 {
        let f = p.enumerate(options(step_size)).unwrap();
        let mut lp = f.into_master(ClpMaster::new().unwrap()).unwrap();
        assert!(lp.solve());
        assert!((lp.objective() - 4.5).abs() < 1e-7);
    }
    // Independent full-route master on an asymmetric heterogeneous fixture.
    p = problem();
    p.capacity = 3.0;
    for i in 0..5 {
        for j in 0..5 {
            p.costs[i][j] = Some(((i * 7 + j * 3 + i * j) % 11 + 1) as f64);
        }
    }
    let mut sp = ClpMaster::new().unwrap();
    for _ in 0..3 {
        sp.add_row(RowSense::Equal, 1.0, &[]).unwrap();
    }
    fn routes(p: &Problem, lp: &mut ClpMaster, order: &mut Vec<usize>, load: f64, cost: f64) {
        let at = *order.last().unwrap();
        if at != 0 {
            let entries: Vec<_> = order[1..].iter().map(|v| (v - 1, 1.0)).collect();
            lp.add_column(cost + p.costs[at][4].unwrap(), &entries)
                .unwrap();
        }
        for v in 1..=3 {
            if order.contains(&v) || load + p.demands[v - 1] > p.capacity {
                continue;
            }
            order.push(v);
            routes(
                p,
                lp,
                order,
                load + p.demands[v - 1],
                cost + p.costs[at][v].unwrap(),
            );
            order.pop();
        }
    }
    routes(&p, &mut sp, &mut vec![0], 0.0, 0.0);
    assert!(sp.solve());
    let mut bounds = Vec::new();
    for size in [1, 2, 4] {
        let mut lp = p
            .enumerate(options(size))
            .unwrap()
            .into_master(ClpMaster::new().unwrap())
            .unwrap();
        assert!(lp.solve());
        bounds.push(lp.objective());
    }
    assert!(bounds[0] <= bounds[1] + 1e-7);
    assert!(bounds[1] <= bounds[2] + 1e-7);
    assert!((bounds[2] - sp.objective()).abs() < 1e-7);

    let mut occupied = ClpMaster::new().unwrap();
    occupied.add_row(RowSense::Equal, 1.0, &[]).unwrap();
    assert!(p
        .enumerate(options(2))
        .unwrap()
        .into_master(occupied)
        .is_err());

    // Missing return arcs make the full model infeasible, even though depot
    // prefixes exist. Their coverage cannot satisfy endpoint conservation.
    for row in &mut p.costs {
        row[4] = None;
    }
    let mut lp = p
        .enumerate(options(4))
        .unwrap()
        .into_master(ClpMaster::new().unwrap())
        .unwrap();
    assert_eq!(
        lp.solve_with_time_limit(std::time::Duration::from_secs(10))
            .unwrap(),
        crate::master::LpSolveOutcome::PrimalInfeasible
    );
}
