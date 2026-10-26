use super::*;
use crate::master::{clp::ClpMaster, LpMaster, RowSense};

fn triangle() -> Problem {
    Problem {
        locations: (0..4)
            .map(|v| Location {
                kind: if v == 0 {
                    LocationKind::Depot
                } else {
                    LocationKind::Customer { demand: 1.0 }
                },
                window: (0.0, 30.0),
                service: 0.0,
            })
            .collect(),
        travel: vec![
            vec![
                Some(Travel {
                    time: 1.0,
                    energy: 1.0,
                    cost: 1.0
                });
                4
            ];
            4
        ],
        depot: 0,
        battery: 10.0,
        load_capacity: 2.0,
    }
}

fn electric() -> Problem {
    let mut p = triangle();
    p.locations[3].kind = LocationKind::Station {
        curve: ChargingCurve::new(vec![(0.0, 0.0), (3.0, 3.0), (6.0, 9.0)]).unwrap(),
    };
    p.locations[0].window.1 = 12.0;
    p.battery = 6.0;
    for (i, row) in p.travel.iter_mut().enumerate() {
        for (j, arc) in row.iter_mut().enumerate() {
            arc.as_mut().unwrap().energy = if (i == 0 || j == 0) && i != 3 && j != 3 {
                4.0
            } else {
                2.0
            };
        }
    }
    p
}

// Independent integer charging-grid witnesses on these integral tiny fixtures.
// Every enumerated order lacking a grid witness is checked against the continuous
// optimizer, so a grid omission cannot silently weaken the comparison pool.
fn tiny_pool(p: &Problem, max_arcs: usize) -> Vec<Vec<Visit>> {
    fn charging(p: &Problem, visits: &mut [Visit], i: usize) -> bool {
        if i == visits.len() {
            return p.replay(visits, 0.0).is_ok();
        }
        if matches!(
            p.locations[visits[i].location].kind,
            LocationKind::Station { .. }
        ) {
            for charge in 0..=p.battery as usize {
                visits[i].recharge = charge as f64;
                if charging(p, visits, i + 1) {
                    return true;
                }
            }
            false
        } else {
            charging(p, visits, i + 1)
        }
    }
    fn orders(p: &Problem, max_arcs: usize, path: &mut Vec<usize>, pool: &mut Vec<Vec<Visit>>) {
        let at = *path.last().unwrap();
        if path.len() > 1
            && path
                .iter()
                .any(|&v| matches!(p.locations[v].kind, LocationKind::Customer { .. }))
            && p.travel[at][p.depot].is_some()
        {
            path.push(p.depot);
            let mut visits = path
                .iter()
                .map(|&location| Visit {
                    location,
                    recharge: 0.0,
                })
                .collect::<Vec<_>>();
            if charging(p, &mut visits, 0) {
                pool.push(visits);
            } else {
                let r = p
                    .optimize_fixed_order(path, 0.0, ChargingOptions::default())
                    .unwrap();
                assert_eq!(
                    r.status,
                    ChargingStatus::Infeasible,
                    "grid missed continuous feasible order {path:?}: {r:?}"
                );
            }
            path.pop();
        }
        if path.len() >= max_arcs {
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
            orders(p, max_arcs, path, pool);
            path.pop();
        }
    }
    let mut pool = Vec::new();
    orders(p, max_arcs, &mut vec![p.depot], &mut pool);
    pool
}

fn pool_lp(p: &Problem, pool: &[Vec<Visit>], fleet: Option<f64>) -> ClpMaster {
    let mut lp = ClpMaster::new().unwrap();
    let rows = p
        .locations
        .iter()
        .map(|loc| {
            if matches!(loc.kind, LocationKind::Customer { .. }) {
                Some(lp.add_row(RowSense::Equal, 1.0, &[]).unwrap())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let fleet = fleet.map(|rhs| lp.add_row(RowSense::LessEqual, rhs, &[]).unwrap());
    for visits in pool {
        let schedule = p.replay(visits, 0.0).unwrap();
        let mut entries = visits
            .iter()
            .filter_map(|v| rows[v.location].map(|r| (r, 1.0)))
            .collect::<Vec<_>>();
        if let Some(row) = fleet {
            entries.push((row, 1.0));
        }
        entries.sort_by_key(|e| e.0);
        lp.add_column(schedule.travel_cost, &entries).unwrap();
    }
    lp.set_tolerances(1e-7, 1e-7).unwrap();
    lp
}

fn backends(max_arcs: usize) -> [EvPricingBackend; 3] {
    let options = DirectOptions {
        max_arcs,
        ..DirectOptions::default()
    };
    [
        EvPricingBackend::PhysicalGraph(PhysicalGraphOptions {
            max_arcs,
            ..PhysicalGraphOptions::default()
        }),
        EvPricingBackend::Direct(options),
        EvPricingBackend::Subpath(SubpathOptions {
            pricing: options,
            ..SubpathOptions::default()
        }),
    ]
}

#[test]
fn both_backends_converge_to_fractional_full_route_pool_lp_without_seeds() {
    let p = triangle();
    let pool = tiny_pool(&p, 4);
    let mut full = pool_lp(&p, &pool, Some(2.0));
    assert!(full.solve());
    assert!((full.objective() - 4.5).abs() < 1e-7);
    for backend in backends(4) {
        let mut master =
            EvMaster::new(p.clone(), ClpMaster::new().unwrap(), 0.0, Some(2.0)).unwrap();
        let result = master
            .optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })
            .unwrap();
        assert_eq!(
            result.status,
            EvColumnGenerationStatus::Converged,
            "{result:?}"
        );
        let snapshot = result.snapshot.unwrap();
        assert_eq!(snapshot.phase, EvMasterPhase::TravelCost);
        assert!(snapshot.artificial_usage <= 1e-7);
        assert!((snapshot.travel_objective.unwrap() - full.objective()).abs() < 1e-7);
        assert!(snapshot
            .route_values
            .iter()
            .any(|v| *v > 1e-7 && (*v - v.round()).abs() > 1e-7));
        assert!(result
            .iterations
            .iter()
            .any(|i| i.phase == EvMasterPhase::Feasibility));
        assert!(result
            .iterations
            .iter()
            .any(|i| i.phase == EvMasterPhase::TravelCost));
        for column in master.columns() {
            p.replay(&column.visits, 0.0).unwrap();
        }
    }
}

#[test]
fn both_backends_match_independently_replayed_electric_route_pool_with_fleet_limit() {
    let p = electric();
    let pool = tiny_pool(&p, 7);
    let mut full = pool_lp(&p, &pool, Some(1.0));
    assert!(full.solve());
    assert!((full.objective() - 5.0).abs() < 1e-7);
    for backend in backends(14) {
        let mut master =
            EvMaster::new(p.clone(), ClpMaster::new().unwrap(), 0.0, Some(1.0)).unwrap();
        let result = master
            .optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })
            .unwrap();
        assert_eq!(
            result.status,
            EvColumnGenerationStatus::Converged,
            "{result:?}"
        );
        let snapshot = result.snapshot.unwrap();
        assert!((snapshot.lp_objective - full.objective()).abs() < 1e-7);
        assert!(master
            .columns()
            .iter()
            .zip(&snapshot.route_values)
            .any(|(c, v)| *v > 1e-7 && c.visits.iter().any(|v| v.recharge > 0.0)));
        for column in master.columns() {
            p.replay(&column.visits, 0.0).unwrap();
        }
    }
}

#[test]
fn complete_phase_one_detects_fleet_and_reachability_infeasibility() {
    let mut unreachable = triangle();
    for row in &mut unreachable.travel {
        row[3] = None;
    }
    for (p, fleet) in [
        (triangle(), Some(1.0)),
        (triangle(), Some(0.0)),
        (unreachable, None),
    ] {
        for backend in backends(4) {
            let mut master =
                EvMaster::new(p.clone(), ClpMaster::new().unwrap(), 0.0, fleet).unwrap();
            let result = master
                .optimize(EvColumnGenerationOptions {
                    backend,
                    ..EvColumnGenerationOptions::default()
                })
                .unwrap();
            assert_eq!(
                result.status,
                EvColumnGenerationStatus::Infeasible,
                "{result:?}"
            );
            assert!(result.snapshot.unwrap().artificial_usage > 1e-7);
            assert_eq!(
                result.iterations.last().unwrap().pricing_status,
                EvPricingStatus::Complete
            );
        }
    }
}

#[test]
fn limited_pricing_never_claims_lp_convergence_or_original_infeasibility() {
    for backend in backends(2) {
        let mut master =
            EvMaster::new(triangle(), ClpMaster::new().unwrap(), 0.0, Some(2.0)).unwrap();
        let result = master
            .optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })
            .unwrap();
        assert_eq!(
            result.status,
            EvColumnGenerationStatus::PricingIncomplete(EvPricingStatus::Bounded),
            "{result:?}"
        );
    }
    for backend in [
        EvPricingBackend::Direct(DirectOptions {
            max_work: 0,
            ..DirectOptions::default()
        }),
        EvPricingBackend::Subpath(SubpathOptions {
            max_subpaths: 0,
            ..SubpathOptions::default()
        }),
    ] {
        let mut master = EvMaster::new(triangle(), ClpMaster::new().unwrap(), 0.0, None).unwrap();
        let result = master
            .optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })
            .unwrap();
        assert_eq!(
            result.status,
            EvColumnGenerationStatus::PricingIncomplete(EvPricingStatus::WorkLimit)
        );
    }
}

#[test]
fn route_insertion_validates_and_deduplicates_and_limits_can_resume() {
    let p = triangle();
    let mut master = EvMaster::new(p, ClpMaster::new().unwrap(), 0.0, None).unwrap();
    assert!(master
        .add_route(&[
            Visit {
                location: 0,
                recharge: 0.0
            },
            Visit {
                location: 1,
                recharge: 3.0
            },
            Visit {
                location: 0,
                recharge: 0.0
            }
        ])
        .is_err());
    let visits = [
        Visit {
            location: 0,
            recharge: 0.0,
        },
        Visit {
            location: 1,
            recharge: 0.0,
        },
        Visit {
            location: 0,
            recharge: 0.0,
        },
    ];
    let column = master.add_route(&visits).unwrap();
    assert_eq!(master.add_route(&visits).unwrap(), column);
    assert_eq!(master.columns().len(), 1);
    let r = master
        .optimize(EvColumnGenerationOptions {
            max_iterations: 0,
            ..EvColumnGenerationOptions::default()
        })
        .unwrap();
    assert_eq!(r.status, EvColumnGenerationStatus::IterationLimit);
    let r = master
        .optimize(EvColumnGenerationOptions {
            time_limit: std::time::Duration::ZERO,
            ..EvColumnGenerationOptions::default()
        })
        .unwrap();
    assert_eq!(r.status, EvColumnGenerationStatus::TimeLimit);
    let r = master
        .optimize(EvColumnGenerationOptions {
            max_iterations: 1,
            ..EvColumnGenerationOptions::default()
        })
        .unwrap();
    assert_eq!(r.status, EvColumnGenerationStatus::IterationLimit);
    assert!(r.snapshot.is_some());
    let r = master
        .optimize(EvColumnGenerationOptions::default())
        .unwrap();
    assert_eq!(r.status, EvColumnGenerationStatus::Converged, "{r:?}");
}

#[test]
fn empty_customer_master_and_invalid_configuration_are_explicit() {
    let mut p = triangle();
    p.locations.truncate(1);
    p.travel = vec![vec![None]];
    for backend in backends(2) {
        let mut m = EvMaster::new(p.clone(), ClpMaster::new().unwrap(), 0.0, Some(0.0)).unwrap();
        let r = m
            .optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })
            .unwrap();
        assert_eq!(r.status, EvColumnGenerationStatus::Converged, "{r:?}");
        assert_eq!(r.snapshot.unwrap().travel_objective, Some(0.0));
    }
    let mut lp = ClpMaster::new().unwrap();
    lp.add_row(RowSense::Equal, 1.0, &[]).unwrap();
    assert!(EvMaster::new(triangle(), lp, 0.0, None).is_err());
    let mut m = EvMaster::new(triangle(), ClpMaster::new().unwrap(), 0.0, None).unwrap();
    assert!(m
        .optimize(EvColumnGenerationOptions {
            lp_tolerance: 0.0,
            ..EvColumnGenerationOptions::default()
        })
        .is_err());
}
