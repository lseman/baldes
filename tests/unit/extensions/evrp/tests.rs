use super::*;

fn curve() -> ChargingCurve {
    ChargingCurve::new(vec![(0.0, 0.0), (5.0, 5.0), (10.0, 15.0)]).unwrap()
}

fn problem() -> Problem {
    let arc = Some(Travel {
        time: 1.0,
        energy: 3.0,
        cost: 2.0,
    });
    Problem {
        locations: vec![
            Location {
                kind: LocationKind::Depot,
                window: (0.0, 100.0),
                service: 0.0,
            },
            Location {
                kind: LocationKind::Station { curve: curve() },
                window: (0.0, 100.0),
                service: 0.0,
            },
            Location {
                kind: LocationKind::Customer { demand: 2.0 },
                window: (10.0, 20.0),
                service: 2.0,
            },
        ],
        travel: vec![vec![arc; 3]; 3],
        depot: 0,
        battery: 10.0,
        load_capacity: 2.0,
    }
}

fn route() -> Vec<Visit> {
    [(0, 0.0), (1, 2.0), (2, 0.0), (1, 1.0), (0, 0.0)]
        .into_iter()
        .map(|(location, recharge)| Visit { location, recharge })
        .collect()
}

#[test]
fn partial_charge_crosses_breakpoint_and_inverts() {
    let c = curve();
    assert_eq!(c.duration(4.0, 7.0).unwrap(), 5.0);
    assert_eq!(c.charge_for(4.0, 5.0).unwrap(), 7.0);
    assert_eq!(c.charge_for(5.0, 0.0).unwrap(), 5.0);
    assert_eq!(c.charge_for(10.0, f64::MAX).unwrap(), 10.0);
    for i in 0..=100 {
        let energy = i as f64 / 10.0;
        let restored = c
            .charge_for(0.0, c.cumulative_time(energy).unwrap())
            .unwrap();
        assert!((restored - energy).abs() < 1e-12);
    }
}

#[test]
fn invalid_curves_and_queries_are_rejected() {
    for knots in [
        vec![],
        vec![(0.0, 0.0)],
        vec![(1.0, 0.0), (10.0, 10.0)],
        vec![(0.0, 0.0), (5.0, 5.0), (5.0, 7.0)],
        vec![(0.0, 0.0), (5.0, 0.0)],
        vec![(0.0, 0.0), (f64::NAN, 2.0)],
    ] {
        assert!(ChargingCurve::new(knots).is_err());
    }
    assert!(curve().duration(8.0, 7.0).is_err());
    assert!(curve().duration(0.0, 11.0).is_err());
    assert!(curve().charge_for(0.0, f64::INFINITY).is_err());
}

#[test]
fn replay_partial_charging_waiting_and_station_revisit() {
    let s = problem().replay(&route(), 0.0).unwrap();
    assert_eq!(s.travel_cost, 8.0);
    assert_eq!(s.delivered, 2.0);
    assert_eq!(s.stops[1].departure, 5.0); // charge from 7 to 9: slow segment
    assert_eq!(s.stops[2].arrival, 6.0);
    assert_eq!(s.stops[2].start, 10.0);
    assert_eq!(s.stops[4].departure_energy, 1.0);
    assert_eq!(s.stops[4].departure, 15.0);
}

#[test]
fn native_pricing_excludes_station_only_routes_and_preserves_positive_columns() {
    let mut p = problem();
    // Station-only return costs four; the cheapest customer route costs four
    // as well. A large fleet reward must never admit an empty customer column.
    for bidirectional in [false, true] {
        let options = NativeOptions {
            bidirectional,
            max_arcs: 4,
            ..NativeOptions::default()
        };
        for fleet in [0., 100.] {
            let r = p.price_native(&[0.; 3], fleet, 0., options).unwrap();
            let route = r.best_route.unwrap();
            assert!(route.charging.visits.iter().any(|v| v.location == 2));
            assert_eq!(route.reduced_cost, 4. - fleet);
            p.replay(&route.charging.visits, 0.).unwrap();
        }
        p.locations[2].kind = LocationKind::Station { curve: curve() };
        let r = p.price_native(&[0.; 3], 100., 0., options).unwrap();
        assert_eq!(r.status, EvPricingStatus::Complete);
        assert!(r.best_route.is_none());
        p.locations[2].kind = LocationKind::Customer { demand: 2. };
    }
}

#[test]
fn native_limits_and_failed_schedule_reconstruction_never_claim_completeness() {
    for options in [
        NativeOptions {
            max_labels: 0,
            ..NativeOptions::default()
        },
        NativeOptions {
            max_front_work: 0,
            ..NativeOptions::default()
        },
        NativeOptions {
            max_reconstruction_work: 0,
            ..NativeOptions::default()
        },
    ] {
        let r = problem()
            .price_native(&[0., 0., 20.], 0., 0., options)
            .unwrap();
        assert_eq!(r.status, EvPricingStatus::WorkLimit, "{r:?}");
    }
}

#[test]
fn native_pricing_rejects_objective_scales_reserved_for_native_sentinels() {
    let mut p = problem();
    for arc in p.travel.iter_mut().flatten().flatten() {
        arc.cost = 1e12;
    }
    let error = p
        .price_native(&[0.; 3], 0., 0., NativeOptions::default())
        .unwrap_err();
    assert!(error.message.contains("rescale cost units"));
}

#[test]
fn native_load_tolerance_matches_exhaustive_replay() {
    let mut p = problem();
    p.locations = (0..6)
        .map(|i| Location {
            kind: if i == 0 {
                LocationKind::Depot
            } else {
                LocationKind::Customer {
                    demand: [0., 0.1, 0.2, 0.3, 0., 0.25][i],
                }
            },
            window: (0., 100.),
            service: 0.,
        })
        .collect();
    p.load_capacity = 0.85;
    p.travel = vec![vec![None; 6]; 6];
    for (u, v) in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 3),
        (3, 2),
        (2, 1),
        (1, 4),
        (4, 5),
        (5, 0),
    ] {
        p.travel[u][v] = Some(Travel {
            time: 0.,
            energy: 0.,
            cost: if (u, v) == (0, 3) { 1. } else { 0. },
        });
    }
    // The cheaper permutation has prefix load .6000000000000001 and final
    // load .8500000000000001, within the 1e-6 capacity tolerance that native
    // pricing, dominance and replay all share.
    let duals = [0., 10., 10., 10., 10., 10.];
    let expected = exhaustive_route_best(&p, &duals, 0., 6);
    assert_eq!(expected, Some(-50.));
    let native = p
        .price_native(
            &duals,
            0.,
            0.,
            NativeOptions {
                max_arcs: 6,
                bidirectional: false,
                ..NativeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(native.status, EvPricingStatus::Complete);
    assert_eq!(native.best_route.unwrap().reduced_cost, expected.unwrap());
}

#[test]
fn replay_rejects_infeasible_schedules() {
    let p = problem();
    let mut r = route();
    r[1].recharge = 0.0;
    r[3].recharge = 0.0;
    assert!(p.replay(&r, 0.0).is_err());
    r = route();
    r[1].recharge = 4.0;
    assert!(p.replay(&r, 0.0).is_err());
    r = route();
    r[2].recharge = 1.0;
    assert!(p.replay(&r, 0.0).is_err());
    r = route();
    r.insert(3, r[2]);
    assert!(p.replay(&r, 0.0).is_err());
    r = route();
    r[1].location = usize::MAX;
    assert!(p.replay(&r, 0.0).is_err());
    let mut p = problem();
    p.load_capacity = 1.0;
    assert!(p.replay(&route(), 0.0).is_err());
    let mut p = problem();
    p.locations[1].window.1 = 4.0;
    assert!(p.replay(&route(), 0.0).is_err());
    let mut p = problem();
    p.travel[0][1] = None;
    assert!(p.replay(&route(), 0.0).is_err());
    assert!(problem().replay(&route(), 30.0).is_err());
}

#[test]
fn station_specific_curves_change_schedule() {
    let mut p = problem();
    p.locations[1].kind = LocationKind::Station {
        curve: ChargingCurve::new(vec![(0.0, 0.0), (10.0, 10.0)]).unwrap(),
    };
    assert_eq!(p.replay(&route(), 0.0).unwrap().stops[1].departure, 3.0);
    p.locations[1].kind = LocationKind::Station {
        curve: ChargingCurve::new(vec![(0.0, 0.0), (9.0, 10.0)]).unwrap(),
    };
    assert!(p.validate().is_err());
}

fn optimized(p: &Problem, order: &[usize]) -> ChargingSolution {
    let r = p
        .optimize_fixed_order(order, 0.0, ChargingOptions::default())
        .unwrap();
    assert_eq!(r.status, ChargingStatus::Optimal, "{r:?}");
    let s = r.solution.unwrap();
    p.replay(&s.visits, 0.0).unwrap();
    s
}

#[test]
fn fixed_order_optimizes_partial_charging_and_reconstructs_revisits() {
    let p = problem();
    let solution = optimized(&p, &[0, 1, 2, 1, 0]);
    assert_eq!(solution.schedule.stops.last().unwrap().departure, 14.0);
    assert_eq!(
        solution.schedule.stops.last().unwrap().departure_energy,
        0.0
    );
    // Only the two energy units needed for the whole route are added.
    assert_eq!(solution.visits.iter().map(|v| v.recharge).sum::<f64>(), 2.0);
    assert!(solution.visits.iter().all(|v| v.recharge >= 0.0));
}

#[test]
fn fixed_order_preserves_infeasibility_unknown_and_invalid_input() {
    let p = problem();
    let r = p
        .optimize_fixed_order(&[0, 1, 2, 1, 0], 0.0, ChargingOptions { max_work: 0 })
        .unwrap();
    assert_eq!(r.status, ChargingStatus::WorkLimit);
    assert!(r.solution.is_none());
    let mut p = problem();
    p.travel[0][1].as_mut().unwrap().energy = 11.0;
    assert_eq!(
        p.optimize_fixed_order(&[0, 1, 0], 0.0, ChargingOptions::default())
            .unwrap()
            .status,
        ChargingStatus::Infeasible
    );
    assert!(p
        .optimize_fixed_order(&[0, 2, 2, 0], 0.0, ChargingOptions::default())
        .is_err());
    assert!(p
        .optimize_fixed_order(&[0, usize::MAX, 0], 0.0, ChargingOptions::default())
        .is_err());
    let mut p = problem();
    p.locations[1].window.1 = 0.0;
    assert_eq!(
        p.optimize_fixed_order(&[0, 1, 0], 0.0, ChargingOptions::default())
            .unwrap()
            .status,
        ChargingStatus::Infeasible
    );
}

#[test]
fn fixed_order_handles_no_stations_and_single_point_windows() {
    let mut p = problem();
    p.locations[2].window = (10.0, 10.0);
    let s = optimized(&p, &[0, 2, 0]);
    assert_eq!(s.schedule.stops[1].start, 10.0);
    assert_eq!(s.schedule.stops[2].departure, 13.0);
    assert_eq!(s.schedule.stops[2].departure_energy, 4.0);
}

#[test]
fn fixed_order_arithmetic_overflow_is_not_infeasibility() {
    let mut p = problem();
    for loc in &mut p.locations {
        loc.window = (0.0, f64::MAX);
    }
    p.travel[0][2].as_mut().unwrap().time = f64::MAX;
    let r = p
        .optimize_fixed_order(&[0, 2, 0], f64::MAX, ChargingOptions::default())
        .unwrap();
    assert_eq!(r.status, ChargingStatus::NumericalFailure);
    assert!(r.solution.is_none());
}

// Independent finite-grid enumeration is an upper bound on continuous optimum.
// These dyadic cases have an optimum on the quarter-energy grid; checking equality
// catches charging allocation and envelope/reconstruction errors without reusing
// the optimizer recurrence.
#[test]
fn fixed_order_matches_exhaustive_grid_with_heterogeneous_and_nonconcave_curves() {
    for (fast, slow) in [(1.0, 2.0), (2.0, 1.0), (0.5, 4.0), (4.0, 0.5)] {
        for opening in [0.0, 8.0, 16.0] {
            let mut p = problem();
            p.battery = 8.0;
            p.locations[1].kind = LocationKind::Station {
                curve: ChargingCurve::new(vec![
                    (0.0, 0.0),
                    (4.0, fast * 4.0),
                    (8.0, fast * 4.0 + slow * 4.0),
                ])
                .unwrap(),
            };
            p.locations[2].window = (opening, 100.0);
            p.locations.push(Location {
                kind: LocationKind::Station {
                    curve: ChargingCurve::new(vec![(0.0, 0.0), (8.0, 8.0)]).unwrap(),
                },
                window: (0.0, 100.0),
                service: 0.0,
            });
            let arc = Some(Travel {
                time: 1.0,
                energy: 3.0,
                cost: 2.0,
            });
            p.travel = vec![vec![arc; 4]; 4];
            let order = [0, 1, 2, 3, 0];
            let solution = optimized(&p, &order);
            let mut grid_best = f64::INFINITY;
            for first in 0..=32 {
                for second in 0..=32 {
                    let mut visits = order.map(|location| Visit {
                        location,
                        recharge: 0.0,
                    });
                    visits[1].recharge = first as f64 / 4.0;
                    visits[3].recharge = second as f64 / 4.0;
                    if let Ok(s) = p.replay(&visits, 0.0) {
                        grid_best = grid_best.min(s.stops.last().unwrap().departure);
                    }
                }
            }
            assert_eq!(
                solution.schedule.stops.last().unwrap().departure,
                grid_best,
                "fast={fast}, slow={slow}, opening={opening}, solution={solution:?}"
            );
        }
    }
}

fn exhaustive_route_best(p: &Problem, duals: &[f64], fleet: f64, arcs: usize) -> Option<f64> {
    fn visit(
        p: &Problem,
        duals: &[f64],
        fleet: f64,
        arcs: usize,
        path: &mut Vec<usize>,
        best: &mut Option<f64>,
    ) {
        let at = *path.last().unwrap();
        if path.len() > 1
            && p.travel[at][p.depot].is_some()
            && path
                .iter()
                .any(|&v| matches!(p.locations[v].kind, LocationKind::Customer { .. }))
        {
            path.push(p.depot);
            let r = p
                .optimize_fixed_order(path, 0.0, ChargingOptions::default())
                .unwrap();
            match r.status {
                ChargingStatus::Optimal => {
                    let s = r.solution.unwrap();
                    let rc = path
                        .iter()
                        .fold(s.schedule.travel_cost - fleet, |c, &v| c - duals[v]);
                    if best.map_or(true, |b| rc < b) {
                        *best = Some(rc);
                    }
                }
                ChargingStatus::Infeasible => {}
                other => panic!("enumeration charging status {other:?}: {path:?}"),
            }
            path.pop();
        }
        if path.len() >= arcs {
            return;
        }
        for next in 0..p.locations.len() {
            if next == at
                || matches!(p.locations[next].kind, LocationKind::Depot)
                || p.travel[at][next].is_none()
                || (matches!(p.locations[next].kind, LocationKind::Customer { .. })
                    && path.contains(&next))
            {
                continue;
            }
            path.push(next);
            visit(p, duals, fleet, arcs, path, best);
            path.pop();
        }
    }
    let mut best = None;
    visit(p, duals, fleet, arcs, &mut vec![p.depot], &mut best);
    best
}

#[test]
fn bidirectional_pricing_matches_exhaustive_route_orders_with_station_repeats() {
    let mut p = problem();
    p.locations.push(Location {
        kind: LocationKind::Station {
            curve: ChargingCurve::new(vec![(0.0, 0.0), (5.0, 10.0), (10.0, 15.0)]).unwrap(),
        },
        window: (0.0, 100.0),
        service: 1.0,
    });
    let arc = Some(Travel {
        time: 1.0,
        energy: 3.0,
        cost: 2.0,
    });
    p.travel = vec![vec![arc; 4]; 4];
    // Force recharging to return from the customer, with multiple station alternatives.
    p.travel[0][2].as_mut().unwrap().energy = 8.0;
    p.travel[2][0].as_mut().unwrap().energy = 8.0;
    for dual in [0.0, 8.0, 20.0] {
        let duals = [0.0, 0.0, dual, 0.0];
        let expected = exhaustive_route_best(&p, &duals, -2.0, 6);
        for dominance in [false, true] {
            let options = DirectOptions {
                max_arcs: 6,
                dominance,
                ..DirectOptions::default()
            };
            let direct = p.price_direct(&duals, -2.0, 0.0, options).unwrap();
            let subpath = p
                .price_subpaths(
                    &duals,
                    -2.0,
                    0.0,
                    SubpathOptions {
                        pricing: options,
                        ..SubpathOptions::default()
                    },
                )
                .unwrap();
            assert!(subpath.statistics.subpaths > 0);
            let native_forward = p
                .price_native(
                    &duals,
                    -2.0,
                    0.0,
                    NativeOptions {
                        max_arcs: 6,
                        bidirectional: false,
                        ..NativeOptions::default()
                    },
                )
                .unwrap();
            let native_backward = p
                .price_native(
                    &duals,
                    -2.0,
                    0.0,
                    NativeOptions {
                        max_arcs: 6,
                        bidirectional: true,
                        ..NativeOptions::default()
                    },
                )
                .unwrap();
            for r in [direct, subpath, native_forward, native_backward] {
                assert_eq!(r.status, EvPricingStatus::Bounded, "{r:?}");
                let route = r.best_route.unwrap();
                p.replay(&route.charging.visits, 0.0).unwrap();
                assert_eq!(Some(route.reduced_cost), expected, "dominance={dominance}");
            }
        }
    }
}

#[test]
fn pricing_covers_elementary_customers_capacity_and_global_status() {
    let mut p = problem();
    p.locations.remove(1);
    p.locations.push(Location {
        kind: LocationKind::Customer { demand: 2.0 },
        window: (0.0, 100.0),
        service: 1.0,
    });
    let arc = Some(Travel {
        time: 0.0,
        energy: 1.0,
        cost: 1.0,
    });
    p.travel = vec![vec![arc; 3]; 3];
    for capacity in [2.0, 4.0] {
        p.load_capacity = capacity;
        let duals = [0.0, 5.0, 8.0];
        let expected = exhaustive_route_best(&p, &duals, 0.0, 3);
        let r = p
            .price_direct(
                &duals,
                0.0,
                0.0,
                DirectOptions {
                    max_arcs: 3,
                    ..DirectOptions::default()
                },
            )
            .unwrap();
        assert_eq!(r.status, EvPricingStatus::Complete);
        assert_eq!(r.best_route.map(|r| r.reduced_cost), expected);
        for bidirectional in [false, true] {
            let native = p
                .price_native(
                    &duals,
                    0.,
                    0.,
                    NativeOptions {
                        max_arcs: 3,
                        bidirectional,
                        ..NativeOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(native.status, EvPricingStatus::Complete, "{native:?}");
            assert_eq!(native.best_route.map(|r| r.reduced_cost), expected);
        }
    }
}

#[test]
fn pricing_limits_and_infeasibility_have_distinct_statuses() {
    let p = problem();
    let duals = [0.0, 0.0, 10.0];
    for (max_work, max_labels) in [(0, 100), (10000, 0)] {
        let r = p
            .price_direct(
                &duals,
                0.0,
                0.0,
                DirectOptions {
                    max_work,
                    max_labels,
                    ..DirectOptions::default()
                },
            )
            .unwrap();
        assert_eq!(r.status, EvPricingStatus::WorkLimit);
    }
    assert!(p
        .price_direct(&[0.0, 1.0, 1.0], 0.0, 0.0, DirectOptions::default())
        .is_err());
    let mut p = problem();
    p.travel[0][2].as_mut().unwrap().energy = 20.0;
    p.travel[1][2].as_mut().unwrap().energy = 20.0;
    let r = p
        .price_direct(
            &duals,
            0.0,
            0.0,
            DirectOptions {
                max_arcs: 4,
                ..DirectOptions::default()
            },
        )
        .unwrap();
    assert_eq!(r.status, EvPricingStatus::Bounded);
    assert!(r.best_route.is_none());
    p.locations[0].window.1 = 3.0;
    let r = p
        .price_direct(
            &duals,
            0.0,
            0.0,
            DirectOptions {
                max_arcs: 4,
                ..DirectOptions::default()
            },
        )
        .unwrap();
    assert_eq!(r.status, EvPricingStatus::Complete);
    assert!(r.best_route.is_none());
}

#[test]
fn pricing_finds_two_chargers_between_depot_and_customer_and_revisits_both() {
    let mut p = problem();
    p.battery = 6.0;
    p.locations[1].kind = LocationKind::Station {
        curve: ChargingCurve::new(vec![(0.0, 0.0), (3.0, 3.0), (6.0, 9.0)]).unwrap(),
    };
    p.locations[2].window = (0.0, 80.0);
    p.locations.push(Location {
        kind: LocationKind::Station {
            curve: ChargingCurve::new(vec![(0.0, 0.0), (6.0, 6.0)]).unwrap(),
        },
        window: (0.0, 90.0),
        service: 1.0,
    });
    p.travel = vec![vec![None; 4]; 4];
    for (i, j) in [(0, 1), (1, 3), (3, 2), (2, 3), (3, 1), (1, 0)] {
        p.travel[i][j] = Some(Travel {
            time: 1.0,
            energy: 3.0,
            cost: 1.0,
        });
    }
    let duals = [0.0, 0.0, 20.0, 0.0];
    let short = p
        .price_direct(
            &duals,
            0.0,
            0.0,
            DirectOptions {
                max_arcs: 5,
                ..DirectOptions::default()
            },
        )
        .unwrap();
    assert_eq!(short.status, EvPricingStatus::Bounded);
    assert!(short.best_route.is_none());
    let full = p
        .price_direct(
            &duals,
            0.0,
            0.0,
            DirectOptions {
                max_arcs: 6,
                ..DirectOptions::default()
            },
        )
        .unwrap();
    assert_eq!(full.status, EvPricingStatus::Bounded, "{full:?}");
    let route = full.best_route.unwrap();
    assert_eq!(route.reduced_cost, -14.0);
    assert_eq!(
        route
            .charging
            .visits
            .iter()
            .map(|v| v.location)
            .collect::<Vec<_>>(),
        vec![0, 1, 3, 2, 3, 1, 0]
    );
    let subpath = p
        .price_subpaths(
            &duals,
            0.0,
            0.0,
            SubpathOptions {
                pricing: DirectOptions {
                    max_arcs: 6,
                    ..DirectOptions::default()
                },
                ..SubpathOptions::default()
            },
        )
        .unwrap();
    assert_eq!(subpath.status, EvPricingStatus::Bounded);
    assert_eq!(subpath.best_route.unwrap().reduced_cost, -14.0);
}

#[test]
fn subpath_limits_and_complete_pricing_match_direct_contract() {
    let mut p = problem();
    let duals = [0.0, 0.0, 10.0];
    let limited = p
        .price_subpaths(
            &duals,
            0.0,
            0.0,
            SubpathOptions {
                max_subpaths: 0,
                ..SubpathOptions::default()
            },
        )
        .unwrap();
    assert_eq!(limited.status, EvPricingStatus::WorkLimit);
    assert!(limited.best_route.is_none());
    for (max_work, max_labels) in [(0, 100), (10000, 0)] {
        let r = p
            .price_subpaths(
                &duals,
                0.0,
                0.0,
                SubpathOptions {
                    pricing: DirectOptions {
                        max_work,
                        max_labels,
                        ..DirectOptions::default()
                    },
                    ..SubpathOptions::default()
                },
            )
            .unwrap();
        assert_eq!(r.status, EvPricingStatus::WorkLimit);
    }
    p.locations[0].window.1 = 14.0;
    let opts = DirectOptions {
        max_arcs: 16,
        ..DirectOptions::default()
    };
    let a = p.price_direct(&duals, 0.0, 0.0, opts).unwrap();
    let b = p
        .price_subpaths(
            &duals,
            0.0,
            0.0,
            SubpathOptions {
                pricing: opts,
                ..SubpathOptions::default()
            },
        )
        .unwrap();
    assert_eq!(a.status, EvPricingStatus::Complete, "{a:?}");
    assert_eq!(b.status, a.status, "{b:?}");
    assert_eq!(
        b.best_route.map(|r| r.reduced_cost),
        a.best_route.map(|r| r.reduced_cost)
    );
}

#[test]
fn both_pricers_match_enumeration_with_two_customers_across_subpaths() {
    let mut p = problem();
    p.battery = 6.0;
    p.locations[0].window.1 = 50.0;
    p.locations[1].kind = LocationKind::Station {
        curve: ChargingCurve::new(vec![(0.0, 0.0), (3.0, 3.0), (6.0, 9.0)]).unwrap(),
    };
    p.locations[2].window = (10.0, 30.0);
    p.locations.push(Location {
        kind: LocationKind::Customer { demand: 2.0 },
        window: (0.0, 20.0),
        service: 1.0,
    });
    p.locations.push(Location {
        kind: LocationKind::Station {
            curve: ChargingCurve::new(vec![(0.0, 0.0), (6.0, 6.0)]).unwrap(),
        },
        window: (0.0, 45.0),
        service: 1.0,
    });
    p.travel = vec![vec![None; 5]; 5];
    for (i, j) in [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 1),
        (4, 0),
        (2, 3),
        (3, 2),
    ] {
        p.travel[i][j] = Some(Travel {
            time: 1.0,
            energy: 2.0,
            cost: if (i, j) == (2, 3) || (i, j) == (3, 2) {
                4.0
            } else {
                1.0
            },
        });
    }
    let duals = [0.0, 0.0, 10.0, 9.0, 0.0];
    for capacity in [2.0, 4.0] {
        p.load_capacity = capacity;
        let expected = exhaustive_route_best(&p, &duals, -1.0, 7);
        for dominance in [false, true] {
            let options = DirectOptions {
                max_arcs: 7,
                dominance,
                ..DirectOptions::default()
            };
            let direct = p.price_direct(&duals, -1.0, 0.0, options).unwrap();
            let subpath = p
                .price_subpaths(
                    &duals,
                    -1.0,
                    0.0,
                    SubpathOptions {
                        pricing: options,
                        ..SubpathOptions::default()
                    },
                )
                .unwrap();
            let native_forward = p
                .price_native(
                    &duals,
                    -1.0,
                    0.0,
                    NativeOptions {
                        max_arcs: 7,
                        bidirectional: false,
                        ..NativeOptions::default()
                    },
                )
                .unwrap();
            let native_backward = p
                .price_native(
                    &duals,
                    -1.0,
                    0.0,
                    NativeOptions {
                        max_arcs: 7,
                        bidirectional: true,
                        ..NativeOptions::default()
                    },
                )
                .unwrap();
            for r in [direct, subpath, native_forward, native_backward] {
                assert_eq!(r.status, EvPricingStatus::Bounded, "{r:?}");
                assert_eq!(
                    r.best_route.map(|r| r.reduced_cost),
                    expected,
                    "capacity={capacity}, dominance={dominance}"
                );
            }
        }
    }
}
