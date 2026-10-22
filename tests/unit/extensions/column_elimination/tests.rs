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
fn time_buckets_preserve_elementary_routes_and_refine_repeated_visits() {
    let mut p = Problem {
        demands: vec![1.0; 20],
        capacity: 20.0,
        costs: vec![vec![Some(1.0); 22]; 22],
        travel_times: vec![vec![8.125; 22]; 22],
        windows: vec![(0.0, 33.0); 22],
        service_times: vec![0.0; 22],
        max_vehicles: None,
    };
    for from in 0..22 {
        for to in 0..22 {
            if (4..=20).contains(&from) || (4..=20).contains(&to) {
                p.costs[from][to] = None;
            }
        }
    }
    let options = Options::default();
    let mut dd = Diagram::time_diagram(&p, &options).unwrap().unwrap();
    let expected = elementary_routes(&p);
    let represented = paths(&dd);
    assert!(represented
        .iter()
        .any(|(_, route)| p.conflict(route).is_some()));
    for route in &expected {
        assert_eq!(
            represented.iter().filter(|(_, path)| path == route).count(),
            1
        );
    }
    for _ in 0..100 {
        let current = paths(&dd);
        let conflict = current
            .iter()
            .find_map(|(_, route)| p.conflict(route).map(|i| route[1..=i + 1].to_vec()));
        if let Some(prefix) = conflict {
            assert_eq!(
                dd.separate_locations(&prefix, &options).unwrap(),
                Some(true)
            );
            dd.reduce();
        } else {
            break;
        }
    }
    let remaining = paths(&dd);
    assert_eq!(remaining.len(), expected.len());
    assert!(remaining
        .iter()
        .all(|(_, route)| p.conflict(route).is_none()));
    p.travel_times[1][2] = 0.0;
    assert!(Diagram::time_diagram(&p, &options).unwrap().is_none());
}

#[test]
fn identical_suffixes_share_nodes_across_visit_layers() {
    let mut p = problem();
    p.capacity = 3.0;
    for customer in 1..=3 {
        p.windows[customer] = (customer as f64, customer as f64);
    }
    let mut dd = Diagram::new(&p, &Options::default()).unwrap();
    assert_eq!(dd.nodes.len(), 5);
    let expected = elementary_routes(&p);
    assert_eq!(paths(&dd).len(), expected.len());
    for route in expected {
        assert_eq!(
            paths(&dd).iter().filter(|(_, path)| path == &route).count(),
            1
        );
    }
    let sizes = (dd.nodes.len(), dd.arcs.len());
    dd.reduce();
    assert_eq!((dd.nodes.len(), dd.arcs.len()), sizes);
}

#[test]
fn backward_deadlines_remove_late_prefixes_and_preserve_indirect_returns() {
    let mut p = problem();
    p.capacity = 3.0;
    p.windows[4].1 = 10.0;
    p.windows[1].0 = 8.0;
    p.windows[3].0 = 9.0;
    p.travel_times[1][4] = 100.0;
    p.travel_times[1][2] = 1.0;
    p.travel_times[2][4] = 1.0;
    let dd = Diagram::new(&p, &Options::default()).unwrap();
    let represented = paths(&dd);
    assert!(represented.iter().any(|(_, path)| path == &[0, 1, 2, 4]));
    assert!(!represented.iter().any(|(_, path)| path == &[0, 1, 3, 2, 4]));
    assert!(!represented.iter().any(|(_, path)| path == &[0, 1, 4]));
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

// An independent route oracle: never uses Diagram to construct its pool.
fn elementary_routes(p: &Problem) -> Vec<Vec<usize>> {
    fn visit(p: &Problem, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if path.len() > 1 {
            path.push(p.demands.len() + 1);
            if p.conflict(path).is_none() {
                out.push(path.clone());
            }
            path.pop();
        }
        for next in 1..=p.demands.len() {
            if !path.contains(&next) {
                path.push(next);
                if p.conflict(path).is_none() {
                    visit(p, path, out);
                }
                path.pop();
            }
        }
    }
    let mut out = Vec::new();
    visit(p, &mut vec![0], &mut out);
    out
}

#[test]
fn equivalent_exact_states_merge_without_losing_routes() {
    let p = Problem {
        demands: vec![1.0; 4],
        capacity: 4.0,
        costs: vec![vec![Some(1.0); 6]; 6],
        travel_times: vec![vec![0.0; 6]; 6],
        windows: vec![(0.0, f64::INFINITY); 6],
        service_times: vec![0.0; 6],
        max_vehicles: None,
    };
    let options = Options::default();
    let mut dd = Diagram::new(&p, &options).unwrap();
    assert_eq!(
        dd.separate_locations(&[1, 2, 3, 1], &options).unwrap(),
        Some(true)
    );
    assert_eq!(
        dd.separate_locations(&[2, 1, 3, 2], &options).unwrap(),
        Some(true)
    );
    let mut incoming = vec![0; dd.nodes.len()];
    for arc in &dd.arcs {
        if arc.active {
            incoming[arc.to] += 1;
        }
    }
    assert!(dd
        .nodes
        .iter()
        .enumerate()
        .any(|(id, node)| node.location == 3 && node.state.is_some() && incoming[id] == 2));
    dd.reduce();
    let remaining = paths(&dd);
    for route in elementary_routes(&p) {
        assert_eq!(remaining.iter().filter(|(_, p)| p == &route).count(), 1);
    }
    // Dead original nodes and disabled arcs are physically reclaimed.
    assert!(dd.arcs.iter().all(|arc| arc.active));
    assert!(dd
        .nodes
        .iter()
        .skip(2)
        .all(|node| !node.outgoing.is_empty()));
}

#[test]
fn capacity_ng_states_preserve_routes_and_handle_zero_demand_by_fallback() {
    let mut p = problem();
    p.capacity = 4.0;
    p.windows.fill((0.0, f64::INFINITY));
    let options = Options::default();
    let mut dd = Diagram::capacity_diagram(&p, &options).unwrap().unwrap();
    for _ in 0..100 {
        let current = paths(&dd);
        for route in elementary_routes(&p) {
            assert_eq!(current.iter().filter(|(_, p)| p == &route).count(), 1);
        }
        let Some(prefix) = current
            .iter()
            .find_map(|(_, route)| p.conflict(route).map(|i| route[1..=i + 1].to_vec()))
        else {
            assert_eq!(current.len(), elementary_routes(&p).len());
            break;
        };
        assert_eq!(
            dd.separate_locations(&prefix, &options).unwrap(),
            Some(true)
        );
        dd.reduce();
    }
    assert!(paths(&dd)
        .iter()
        .all(|(_, route)| p.conflict(route).is_none()));
    // Exact GCD scaling also preserves fractional capacity thresholds.
    p.demands = vec![2.0, 4.0, 6.0];
    p.capacity = 10.5;
    let scaled = Diagram::capacity_diagram(&p, &options).unwrap().unwrap();
    let remaining = paths(&scaled);
    for route in elementary_routes(&p) {
        assert_eq!(remaining.iter().filter(|(_, p)| p == &route).count(), 1);
    }
    p.demands[0] = 0.0;
    assert!(Diagram::capacity_diagram(&p, &options).unwrap().is_none());
    p.demands[0] = 0.5;
    assert!(Diagram::capacity_diagram(&p, &options).unwrap().is_none());
}

#[test]
fn repeated_reduction_and_refinement_match_independent_route_enumeration() {
    for seed in 0..32 {
        let mut p = problem();
        p.capacity = 2.0 + (seed % 2) as f64;
        p.demands[0] = if seed % 3 == 0 { 0.0 } else { 0.1 };
        p.demands[1] = 0.2;
        p.windows[1] = (2.0, 8.0);
        p.windows[2] = (0.0, 7.0);
        p.windows[4] = (0.0, 12.0);
        for i in 0..5 {
            for j in 0..5 {
                p.travel_times[i][j] = ((i * 7 + j * 3 + seed) % 9) as f64 / 2.0;
            }
        }
        if seed % 4 == 0 {
            p.costs[1][2] = None;
        }
        let expected = elementary_routes(&p);
        let mut dd = Diagram::new(&p, &Options::default()).unwrap();
        let mut converged = false;
        for _ in 0..100 {
            let current = paths(&dd);
            for route in &expected {
                assert_eq!(
                    current.iter().filter(|(_, p)| p == route).count(),
                    1,
                    "seed {seed}"
                );
            }
            let conflict = current
                .iter()
                .find_map(|(_, route)| p.conflict(route).map(|i| route[1..=i + 1].to_vec()));
            if let Some(prefix) = conflict {
                assert_eq!(
                    dd.separate_locations(&prefix, &Options::default()).unwrap(),
                    Some(true)
                );
                dd.reduce();
            } else {
                assert_eq!(current.len(), expected.len(), "seed {seed}");
                converged = true;
                break;
            }
        }
        assert!(converged, "seed {seed}");
        let before = paths(&dd);
        dd.reduce();
        assert_eq!(paths(&dd), before);
    }
}

#[test]
fn resource_reduction_respects_floating_point_feasibility_boundaries() {
    for capacity in [0.3, 0.1 + 0.2] {
        let mut p = problem();
        p.demands = vec![0.1, 0.2, 0.0];
        p.capacity = capacity;
        p.travel_times = vec![vec![0.1; 5]; 5];
        p.windows[4].1 = 0.1 + 0.2;
        let dd = Diagram::new(&p, &Options::default()).unwrap();
        let remaining = paths(&dd);
        for route in elementary_routes(&p) {
            assert_eq!(remaining.iter().filter(|(_, p)| p == &route).count(), 1);
        }
    }
}

#[test]
fn inactive_time_is_omitted_only_when_elementary_routes_cannot_overflow() {
    use super::state::ExactState;
    let mut p = problem();
    p.capacity = 3.0;
    p.windows.fill((0.0, f64::INFINITY));
    p.travel_times[0][2] = 2.0;
    let reach = |p: &Problem, route: &[usize]| {
        route.windows(2).fold(ExactState::root(p), |state, pair| {
            state.advance(p, pair[0], pair[1]).unwrap()
        })
    };
    assert_eq!(reach(&p, &[0, 1, 2, 3]), reach(&p, &[0, 2, 1, 3]));
    p.windows[3].1 = 100.0;
    assert_ne!(reach(&p, &[0, 1, 2, 3]), reach(&p, &[0, 2, 1, 3]));
    p.windows[3].1 = f64::INFINITY;
    p.travel_times = vec![vec![1e308; 5]; 5];
    let first = ExactState::root(&p).advance(&p, 0, 1).unwrap();
    assert!(first.advance(&p, 1, 2).is_none());
}

#[test]
fn refinement_preserves_all_feasible_routes() {
    let mut p = problem();
    p.capacity = 3.0;
    let options = Options::default();
    let mut dd = Diagram::new(&p, &options).unwrap();
    let feasible: Vec<_> = paths(&dd)
        .into_iter()
        .filter(|(_, r)| p.conflict(r).is_none())
        .map(|(_, r)| r)
        .collect();
    assert_eq!(feasible.len(), 15);
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
fn initial_pruning_preserves_feasible_routes_with_directed_nonmetric_times() {
    for seed in 0..12 {
        let mut p = problem();
        p.demands[0] = 0.0;
        p.demands[2] = if seed % 2 == 0 { 3.0 } else { 1.0 };
        p.windows[1] = (2.0, 4.0);
        p.windows[2] = (0.0, 5.0);
        p.windows[4] = (0.0, 7.0);
        p.service_times[1] = 1.0;
        for i in 0..5 {
            for j in 0..5 {
                p.travel_times[i][j] = ((i * 7 + j * 3 + seed) % 9) as f64;
            }
        }
        // Construct the unpruned graph independently of p's resources.
        let mut relaxed = p.clone();
        relaxed.demands.fill(0.0);
        relaxed.windows.fill((0.0, f64::INFINITY));
        let original = Diagram::new(&relaxed, &Options::default()).unwrap();
        let pruned = Diagram::new(&p, &Options::default()).unwrap();
        assert!(pruned.arcs.len() < original.arcs.len());
        let remaining = paths(&pruned);
        for (_, route) in paths(&original) {
            if p.conflict(&route).is_none() {
                assert_eq!(remaining.iter().filter(|(_, p)| p == &route).count(), 1);
            }
        }
    }
}

#[test]
fn batched_refinement_resolves_shared_prefixes_and_preserves_feasible_routes() {
    let mut p = problem();
    p.capacity = 3.0;
    let options = Options::default();
    let mut dd = Diagram::new(&p, &options).unwrap();
    let original = paths(&dd);
    let feasible: Vec<_> = original
        .iter()
        .filter(|(_, route)| p.conflict(route).is_none())
        .map(|(_, route)| route.clone())
        .collect();
    let conflicts: Vec<_> = original
        .iter()
        .filter_map(|(_, route)| p.conflict(route).map(|i| route[1..=i + 1].to_vec()))
        .collect();
    assert!(conflicts.len() > 1);
    for prefix in &conflicts {
        assert_ne!(
            dd.separate_locations(prefix, &options).unwrap(),
            Some(false)
        );
    }
    let remaining = paths(&dd);
    assert_eq!(remaining.len(), feasible.len());
    for route in &feasible {
        assert_eq!(remaining.iter().filter(|(_, p)| p == route).count(), 1);
    }
    // Replaying a stale conflict is harmless, including shared prefixes.
    for prefix in &conflicts {
        assert_eq!(dd.separate_locations(prefix, &options).unwrap(), None);
    }
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
    let mut p = problem();
    p.capacity = 3.0;
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
        let mut lp = ClpMaster::new().unwrap();
        for _ in &p.demands {
            lp.add_row(RowSense::Equal, 1.0, &[]).unwrap();
        }
        if let Some(k) = p.max_vehicles {
            lp.add_row(RowSense::LessEqual, k, &[]).unwrap();
        }
        for path in elementary_routes(p) {
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
        assert!(r.routes.iter().any(|r| (r.flow - 0.5).abs() < 1e-6));
        assert!((full_route_lp(&p).unwrap() - r.lower_bound.unwrap()).abs() < 1e-6);
    }

    #[test]
    fn capacity_ng_auto_mode_matches_analytical_fractional_bound() {
        let p = Problem {
            demands: vec![1.0; 20],
            capacity: 3.0,
            costs: vec![vec![Some(1.0); 22]; 22],
            travel_times: vec![vec![0.0; 22]; 22],
            windows: vec![(0.0, f64::INFINITY); 22],
            service_times: vec![0.0; 22],
            max_vehicles: None,
        };
        let result = solve(&p, Options::default());
        assert_eq!(result.status, Status::Converged);
        // At most three customers per route: total route flow >=20/3.
        // All triples cost four; symmetric fractional triples attain this.
        assert!((result.lower_bound.unwrap() - 80.0 / 3.0).abs() < 1e-6);
        assert!(result
            .routes
            .iter()
            .all(|route| p.conflict(&route.path).is_none()));
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
        let mut p = problem();
        p.capacity = 3.0;
        p.costs[1][2] = Some(0.0);
        p.costs[2][1] = Some(0.0);
        for customer in 1..=2 {
            p.costs[customer][3] = Some(100.0);
            p.costs[3][customer] = Some(100.0);
        }
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
