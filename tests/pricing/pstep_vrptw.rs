use baldes::extensions::pstep::*;
use baldes::master::{LpMaster, RowSense};
use baldes::Result;

fn problem() -> TimeWindowProblem {
    TimeWindowProblem {
        routing: Problem {
            demands: vec![1.0, 1.0, 1.0],
            capacity: 3.0,
            costs: vec![vec![Some(1.0); 5]; 5],
        },
        windows: vec![
            (0.0, 100.0),
            (2.0, 5.0),
            (10.0, 13.0),
            (6.0, 8.0),
            (0.0, 100.0),
        ],
        service: vec![0.0, 2.0, 2.0, 2.0, 0.0],
    }
}
fn replay(problem: &TimeWindowProblem, step: &TimedStep) {
    let mut time = step.start_time;
    for arc in step.step.path.windows(2) {
        assert!(
            time >= problem.windows[arc[0]].0 - 1e-8 && time <= problem.windows[arc[0]].1 + 1e-8
        );
        time = (time + problem.service[arc[0]] + problem.routing.costs[arc[0]][arc[1]].unwrap())
            .max(problem.windows[arc[1]].0);
        assert!(time <= problem.windows[arc[1]].1 + 1e-8);
    }
    assert!((time - step.end_time).abs() < 1e-8);
}
#[test]
fn waiting_frontier_contains_kink_and_latest_start() -> Result<()> {
    let problem = TimeWindowProblem {
        routing: Problem {
            demands: vec![1.0, 1.0],
            capacity: 2.0,
            costs: vec![vec![Some(1.0); 4]; 4],
        },
        windows: vec![(0.0, 100.0), (0.0, 20.0), (10.0, 30.0), (0.0, 100.0)],
        service: vec![0.0; 4],
    };
    let pool = problem.enumerate(EnumerationOptions {
        p: 1,
        ..Default::default()
    })?;
    let times: Vec<_> = pool
        .steps()
        .iter()
        .filter(|s| s.step.path == vec![1, 2])
        .map(|s| (s.start_time, s.end_time))
        .collect();
    assert!(times.contains(&(9.0, 10.0)));
    assert!(times.contains(&(20.0, 21.0)));
    for step in pool.steps() {
        replay(&problem, step);
    }
    Ok(())
}
#[test]
fn timed_pricing_matches_full_pool_minimum_and_replays_schedules() -> Result<()> {
    let mut problem = problem();
    problem.routing.costs[3][1] = None;
    let n = problem.routing.demands.len();
    for p in 1..=n + 1 {
        let pool = problem.enumerate(EnumerationOptions {
            p,
            ..Default::default()
        })?;
        for sample in 0..4 {
            let duals: Vec<_> = (0..4 * n)
                .map(|i| (((i * 17 + sample * 11) % 23) as f64 - 11.0) * 0.31)
                .collect();
            for weight in [0.0, 1.0] {
                let expected = pool
                    .steps()
                    .iter()
                    .map(|s| {
                        weight * s.step.cost
                            - problem
                                .entries(s)
                                .iter()
                                .map(|&(r, a)| a * duals[r])
                                .sum::<f64>()
                    })
                    .reduce(f64::min);
                let options = PricingOptions {
                    p,
                    method: PricingMethod::Enumeration,
                    ..Default::default()
                };
                let cpu = problem.price_steps(&duals, weight, &options)?;
                assert_eq!(cpu.minimum_reduced_cost.is_some(), expected.is_some());
                if let Some(expected) = expected {
                    assert!((cpu.minimum_reduced_cost.unwrap() - expected).abs() < 1e-8);
                }
                for step in &cpu.steps {
                    replay(&problem, step);
                }
                #[cfg(feature = "cuda")]
                {
                    let gpu = problem.price_steps(
                        &duals,
                        weight,
                        &PricingOptions {
                            method: PricingMethod::Cuda,
                            ..options
                        },
                    )?;
                    assert_eq!(cpu.enumerated_extensions, gpu.enumerated_extensions);
                    assert_eq!(cpu.steps.len(), gpu.steps.len());
                    assert_eq!(
                        cpu.minimum_reduced_cost.is_some(),
                        gpu.minimum_reduced_cost.is_some()
                    );
                    if let Some(expected) = expected {
                        assert!((gpu.minimum_reduced_cost.unwrap() - expected).abs() < 1e-8);
                    }
                    for (a, b) in cpu.steps.iter().zip(&gpu.steps) {
                        replay(&problem, b);
                        let rc = |s: &TimedStep| {
                            weight * s.step.cost
                                - problem
                                    .entries(s)
                                    .iter()
                                    .map(|&(r, a)| a * duals[r])
                                    .sum::<f64>()
                        };
                        assert!((rc(a) - rc(b)).abs() < 1e-8);
                    }
                }
            }
        }
    }
    Ok(())
}
#[cfg(feature = "clp")]
#[test]
fn vrptw_cg_matches_full_pool_and_independent_route_lp() -> Result<()> {
    use baldes::master::clp::ClpMaster;
    // Both customers can be reached separately at time 1; neither can follow
    // the other. CVRP permits one route (cost 3), VRPTW needs two (cost 4).
    let problem = TimeWindowProblem {
        routing: Problem {
            demands: vec![1.0, 1.0],
            capacity: 2.0,
            costs: vec![vec![Some(1.0); 4]; 4],
        },
        windows: vec![(0.0, 10.0), (1.0, 1.0), (1.0, 1.0), (0.0, 10.0)],
        service: vec![0.0, 1.0, 1.0, 0.0],
    };
    for p in 1..=3 {
        let pool = problem.enumerate(EnumerationOptions {
            p,
            ..Default::default()
        })?;
        let mut lp = pool.into_master(ClpMaster::new()?)?;
        assert!(lp.solve());
        assert!((lp.objective() - 4.0).abs() < 1e-7);
        let mut methods = vec![PricingMethod::Enumeration];
        #[cfg(feature = "cuda")]
        methods.push(PricingMethod::Cuda);
        for method in methods {
            let result = problem.solve_column_generation(
                ClpMaster::new()?,
                ColumnGenerationOptions {
                    pricing: PricingOptions {
                        p,
                        method,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )?;
            assert_eq!(result.status, ColumnGenerationStatus::Converged);
            assert!((result.lp_bound.unwrap() - lp.objective()).abs() < 1e-7);
        }
    }
    // Independent full-route set partitioning, propagating time forward.
    let problem = self::problem();
    let n = problem.routing.demands.len();
    let mut sp = ClpMaster::new()?;
    for _ in 0..n {
        sp.add_row(RowSense::Equal, 1.0, &[])?;
    }
    fn routes(
        problem: &TimeWindowProblem,
        sp: &mut ClpMaster,
        path: &mut Vec<usize>,
        load: f64,
        time: f64,
        cost: f64,
    ) -> Result<()> {
        let n = problem.routing.demands.len();
        let tail = *path.last().unwrap();
        for head in 1..=n + 1 {
            if path.contains(&head) || (tail == 0 && head == n + 1) {
                continue;
            }
            let Some(arc) = problem.routing.costs[tail][head] else {
                continue;
            };
            let load = load
                + if head <= n {
                    problem.routing.demands[head - 1]
                } else {
                    0.0
                };
            let time = (time + problem.service[tail] + arc).max(problem.windows[head].0);
            if load > problem.routing.capacity || time > problem.windows[head].1 {
                continue;
            }
            if head == n + 1 {
                let entries: Vec<_> = path[1..].iter().map(|&v| (v - 1, 1.0)).collect();
                sp.add_column(cost + arc, &entries)?;
            } else {
                path.push(head);
                routes(problem, sp, path, load, time, cost + arc)?;
                path.pop();
            }
        }
        Ok(())
    }
    routes(
        &problem,
        &mut sp,
        &mut vec![0],
        0.0,
        problem.windows[0].0,
        0.0,
    )?;
    assert!(sp.solve());
    let result = problem.solve_column_generation(
        ClpMaster::new()?,
        ColumnGenerationOptions {
            pricing: PricingOptions {
                p: n + 1,
                method: PricingMethod::Enumeration,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert_eq!(result.status, ColumnGenerationStatus::Converged);
    assert!((result.lp_bound.unwrap() - sp.objective()).abs() < 1e-7);
    Ok(())
}

#[test]
fn vrptw_validates_resources_and_honors_limits() -> Result<()> {
    let mut instance = problem();
    instance.windows[1].1 = f64::INFINITY;
    assert!(instance.validate().is_err());
    instance = problem();
    instance.service[1] = -1.0;
    assert!(instance.validate().is_err());
    instance = problem();
    assert!(instance
        .enumerate(EnumerationOptions {
            p: 2,
            max_columns: 1,
            ..Default::default()
        })
        .is_err());
    let options = PricingOptions {
        p: 2,
        time_limit: std::time::Duration::ZERO,
        ..Default::default()
    };
    assert!(instance.price_steps(&vec![0.0; 12], 1.0, &options).is_err());
    let options = PricingOptions {
        p: 2,
        max_labels_per_job: 1,
        ..Default::default()
    };
    assert!(instance.price_steps(&vec![0.0; 12], 1.0, &options).is_err());
    #[cfg(feature = "cuda")]
    assert!(instance
        .price_steps(
            &vec![0.0; 12],
            1.0,
            &PricingOptions {
                method: PricingMethod::Cuda,
                ..options
            }
        )
        .is_err());
    Ok(())
}

#[cfg(feature = "clp")]
#[test]
fn cvrp_column_generation_still_matches_full_pool() -> Result<()> {
    use baldes::master::clp::ClpMaster;
    let problem = problem().routing;
    for p in 1..=4 {
        let pool = problem.enumerate(EnumerationOptions {
            p,
            ..Default::default()
        })?;
        let mut lp = pool.into_master(ClpMaster::new()?)?;
        assert!(lp.solve());
        let mut methods = vec![PricingMethod::Enumeration];
        #[cfg(feature = "cuda")]
        methods.push(PricingMethod::Cuda);
        for method in methods {
            let result = problem.solve_column_generation(
                ClpMaster::new()?,
                ColumnGenerationOptions {
                    pricing: PricingOptions {
                        p,
                        method,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )?;
            assert_eq!(result.status, ColumnGenerationStatus::Converged);
            assert!((result.lp_bound.unwrap() - lp.objective()).abs() < 1e-7);
        }
    }
    Ok(())
}

#[cfg(all(feature = "clp", feature = "cuda"))]
#[test]
fn gpu_cli_selects_cuda_and_enforces_solomon_windows() {
    let directory =
        std::env::temp_dir().join(format!("baldes-pstep-vrptw-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("tiny.txt");
    let output = directory.join("result.json");
    std::fs::write(&input,"tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\nCUST NO. XCOORD YCOORD DEMAND READY DUE SERVICE\n0 0 0 0 0 10 0\n1 1 0 1 1 1 1\n2 -1 0 1 1 1 1\n").unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_baldes"))
        .arg("pstep")
        .arg(&input)
        .args([
            "--p",
            "2",
            "--gpu",
            "--rounding",
            "none",
            "--enumerate",
            "--json",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let result: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(result["time_windows_enforced"], true);
    assert_eq!(result["pricing_method"], "Cuda");
    assert!((result["lp_bound"].as_f64().unwrap() - 4.0).abs() < 1e-7);
    assert!((result["enumerated_bound"].as_f64().unwrap() - 4.0).abs() < 1e-7);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
