use super::*;
use diagram::Diagram;

fn problem() -> Problem {
    Problem {
        demands: vec![1.0; 3],
        capacity: 2.0,
        costs: vec![vec![Some(1.0); 5]; 5],
        travel_times: vec![vec![1.0; 5]; 5],
        windows: vec![(0.0, 100.0); 5],
        service_times: vec![0.0; 5],
        max_vehicles: None,
    }
}

#[test]
fn instance_adapter_preserves_fleet_distances_and_unbounded_windows() {
    use crate::inputs::{Customer, Instance, Rounding};
    let customer = |demand| Customer {
        x: 0.0,
        y: 0.0,
        demand,
        ready: 0.0,
        due: f64::INFINITY,
        service: 0.0,
    };
    let instance = Instance {
        name: "directed".into(),
        vehicles: 2,
        capacity: 4.0,
        customers: vec![customer(0.0), customer(1.0), customer(2.0)],
        rounding: Rounding::Nearest,
        distances: Some(vec![
            vec![0.0, 2.0, 3.0],
            vec![4.0, 0.0, 5.0],
            vec![6.0, 7.0, 0.0],
        ]),
    };
    let p = Problem::from_instance(&instance).unwrap();
    assert_eq!(p.demands, vec![1.0, 2.0]);
    assert_eq!(p.max_vehicles, Some(2.0));
    assert_eq!(p.costs[1][3], Some(4.0));
    assert_eq!(p.travel_times[3][2], 3.0);
    assert_eq!(p.conflict(&[0, 1, 2, 3]), None);
    let mut bad = instance;
    bad.distances.as_mut().unwrap().pop();
    assert!(Problem::from_instance(&bad).is_err());
}

fn paths(dd: &Diagram) -> Vec<(Vec<usize>, Vec<usize>)> {
    fn visit(
        dd: &Diagram,
        at: usize,
        ids: Vec<usize>,
        path: Vec<usize>,
        out: &mut Vec<(Vec<usize>, Vec<usize>)>,
    ) {
        if at == 1 {
            out.push((ids, path));
            return;
        }
        for &id in &dd.nodes[at].outgoing {
            if !dd.arcs[id].active {
                continue;
            }
            let to = dd.arcs[id].to;
            let mut ids = ids.clone();
            ids.push(id);
            let mut path = path.clone();
            path.push(dd.nodes[to].location);
            visit(dd, to, ids, path, out);
        }
    }
    let mut out = vec![];
    visit(dd, 0, vec![], vec![0], &mut out);
    out
}

#[test]
fn refinement_preserves_all_feasible_routes() {
    let p = problem();
    let options = Options::default();
    let mut dd = Diagram::new(&p, &options).unwrap();
    let feasible: Vec<_> = paths(&dd)
        .into_iter()
        .filter(|(_, r)| p.conflict(r).is_none())
        .map(|(_, r)| r)
        .collect();
    assert_eq!(feasible.len(), 9);
    let mut refinements = 0;
    loop {
        let all = paths(&dd);
        for r in &feasible {
            assert_eq!(all.iter().filter(|(_, p)| p == r).count(), 1);
        }
        let bad = all
            .iter()
            .find_map(|(ids, r)| p.conflict(r).map(|i| ids[..=i].to_vec()));
        let Some(prefix) = bad else {
            break;
        };
        assert!(dd.separate(&prefix, &options).unwrap());
        assert!(paths(&dd).len() < all.len());
        refinements += 1;
        assert!(refinements < 100);
    }
    assert!(refinements > 0);
    assert_eq!(paths(&dd).len(), feasible.len());
}

#[test]
fn replay_checks_capacity_elementarity_waiting_service_and_return() {
    let mut p = problem();
    assert_eq!(p.conflict(&[0, 1, 2, 3, 4]), Some(2));
    p.capacity = 4.0;
    p.demands[0] = 0.0;
    assert_eq!(p.conflict(&[0, 1, 2, 1, 4]), Some(2));
    p.windows[1] = (10.0, 10.0);
    p.service_times[1] = 5.0;
    p.windows[2] = (0.0, 15.0);
    assert_eq!(p.conflict(&[0, 1, 2, 4]), Some(1));
    p.windows[2].1 = 16.0;
    assert_eq!(p.conflict(&[0, 1, 2, 4]), None);
    p.windows[4].1 = 16.0;
    assert_eq!(p.conflict(&[0, 1, 2, 4]), Some(2));
    p.costs[0][1] = None;
    assert_eq!(p.conflict(&[0, 1, 4]), Some(0));
}

#[test]
fn limits_are_atomic_and_data_is_validated() {
    let p = problem();
    let mut options = Options::default();
    let mut dd = Diagram::new(&p, &options).unwrap();
    let all = paths(&dd);
    let prefix = all
        .iter()
        .find_map(|(ids, r)| p.conflict(r).map(|i| ids[..=i].to_vec()))
        .unwrap();
    options.max_nodes = dd.nodes.len();
    assert!(!dd.separate(&prefix, &options).unwrap());
    assert_eq!(paths(&dd), all);
    options.max_nodes = 10000;
    options.max_arcs = dd.arcs.len();
    assert!(!dd.separate(&prefix, &options).unwrap());
    assert_eq!(paths(&dd), all);
    assert!(Diagram::new(
        &p,
        &Options {
            max_nodes: 2,
            ..Default::default()
        }
    )
    .is_err());
    assert!(Diagram::new(
        &p,
        &Options {
            max_arcs: 1,
            ..Default::default()
        }
    )
    .is_err());
    let mut p = p;
    p.travel_times[0][1] = f64::NAN;
    assert!(p.validate().is_err());
}

#[cfg(feature = "clp")]
mod lp {
    use super::*;
    use crate::master::solvers::ClpMaster;
    use crate::master::{LpMaster, RowSense};

    fn solve(p: &Problem, options: Options) -> SolveResult {
        p.solve_column_elimination(ClpMaster::new, options).unwrap()
    }

    #[test]
    fn singleton_missing_return_and_nonempty_backend() {
        let mut p = Problem {
            demands: vec![0.0],
            capacity: 1.0,
            costs: vec![vec![Some(1.0); 3]; 3],
            travel_times: vec![vec![1.0; 3]; 3],
            windows: vec![(0.0, f64::INFINITY); 3],
            service_times: vec![0.0; 3],
            max_vehicles: Some(1.0),
        };
        let r = solve(&p, Options::default());
        assert_eq!(r.status, Status::Converged);
        assert_eq!(r.routes[0].path, vec![0, 1, 2]);
        assert!((r.lower_bound.unwrap() - 2.0).abs() < 1e-6);
        assert!(p
            .solve_column_elimination(
                || {
                    let mut lp = ClpMaster::new()?;
                    lp.add_row(RowSense::Equal, 0.0, &[])?;
                    Ok(lp)
                },
                Options::default()
            )
            .is_err());
        p.costs[1][2] = None;
        assert_eq!(solve(&p, Options::default()).status, Status::Infeasible);
    }

    fn full_route_lp(p: &Problem) -> Option<f64> {
        let dd = Diagram::new(p, &Options::default()).unwrap();
        let mut lp = ClpMaster::new().unwrap();
        for _ in &p.demands {
            lp.add_row(RowSense::Equal, 1.0, &[]).unwrap();
        }
        if let Some(k) = p.max_vehicles {
            lp.add_row(RowSense::LessEqual, k, &[]).unwrap();
        }
        for (_, path) in paths(&dd) {
            if p.conflict(&path).is_some() {
                continue;
            }
            let mut entries: Vec<_> = path[1..path.len() - 1]
                .iter()
                .map(|v| (v - 1, 1.0))
                .collect();
            if p.max_vehicles.is_some() {
                entries.push((p.demands.len(), 1.0));
            }
            entries.sort_unstable_by_key(|e| e.0);
            let cost = path.windows(2).map(|w| p.costs[w[0]][w[1]].unwrap()).sum();
            let c = lp.add_column(cost, &entries).unwrap();
            lp.set_upper(c, 1.0).unwrap();
        }
        if lp.solve() {
            Some(lp.objective())
        } else {
            None
        }
    }

    #[test]
    fn fractional_lp_matches_full_route_enumeration() {
        let p = problem();
        let r = solve(&p, Options::default());
        assert_eq!(r.status, Status::Converged);
        assert!((r.lower_bound.unwrap() - 4.5).abs() < 1e-6);
        assert!(r.refinements > 0);
        assert!(r.routes.iter().any(|r| (r.flow - 0.5).abs() < 1e-6));
        assert!((full_route_lp(&p).unwrap() - r.lower_bound.unwrap()).abs() < 1e-6);
    }

    #[test]
    fn asymmetric_windows_and_fleet_match_full_pool() {
        for seed in 0..12usize {
            let mut p = problem();
            p.capacity = 2.0 + (seed % 2) as f64;
            for i in 0..5 {
                for j in 0..5 {
                    p.costs[i][j] = Some(1.0 + ((i * 7 + j * 3 + seed) % 9) as f64);
                    p.travel_times[i][j] = 1.0 + ((i + j) % 3) as f64;
                }
            }
            p.windows[1] = (2.0, 4.0);
            p.windows[2] = (0.0, 5.0);
            p.windows[4] = (0.0, 9.0);
            p.service_times[3] = 1.0;
            p.costs[1][2] = None;
            p.max_vehicles = Some(2.0);
            let expected = full_route_lp(&p);
            let r = solve(&p, Options::default());
            match expected {
                Some(bound) => {
                    assert_eq!(r.status, Status::Converged, "seed {seed}");
                    assert!((bound - r.lower_bound.unwrap()).abs() < 1e-6, "seed {seed}");
                    assert!(r.routes.iter().all(|r| p.conflict(&r.path).is_none()));
                }
                None => assert_eq!(r.status, Status::Infeasible),
            }
        }
    }

    #[test]
    fn infeasibility_and_stop_statuses() {
        let mut p = problem();
        p.demands[0] = 3.0;
        assert_eq!(solve(&p, Options::default()).status, Status::Infeasible);
        let mut p = problem();
        p.max_vehicles = Some(1.0);
        assert_eq!(solve(&p, Options::default()).status, Status::Infeasible);
        let p = problem();
        let r = solve(
            &p,
            Options {
                max_iterations: 0,
                ..Default::default()
            },
        );
        assert_eq!(r.status, Status::IterationLimit);
        assert!(r.lower_bound.is_none());
        assert_eq!(
            solve(
                &p,
                Options {
                    time_limit: Duration::ZERO,
                    ..Default::default()
                }
            )
            .status,
            Status::TimeLimit
        );
        let r = solve(
            &p,
            Options {
                max_iterations: 1,
                ..Default::default()
            },
        );
        assert_eq!(r.status, Status::IterationLimit);
        assert!(r.lower_bound.is_some());
        assert!(r.routes.is_empty());
        let r = solve(
            &p,
            Options {
                max_nodes: 11,
                ..Default::default()
            },
        );
        assert_eq!(r.status, Status::DiagramLimit);
        assert!(r.routes.is_empty());
    }
}
