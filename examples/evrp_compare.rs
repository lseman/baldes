//! Reproducible tiny reference comparisons, not a published-instance benchmark.
#[cfg(feature = "clp")]
fn main() -> baldes::Result<()> {
    use baldes::extensions::evrp::*;
    use baldes::master::clp::ClpMaster;
    println!(
        "case,backend,status,lp_objective,expected_lp,columns,pricing_calls,labels,subpaths,work"
    );
    for electric in [false, true] {
        let mut problem = Problem {
            locations: (0..4)
                .map(|i| Location {
                    kind: if i == 0 {
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
        };
        if electric {
            problem.battery = 6.0;
            problem.locations[0].window.1 = 12.0;
            problem.locations[3].kind = LocationKind::Station {
                curve: ChargingCurve::new(vec![(0.0, 0.0), (3.0, 3.0), (6.0, 9.0)])?,
            };
            for (i, row) in problem.travel.iter_mut().enumerate() {
                for (j, arc) in row.iter_mut().enumerate() {
                    arc.as_mut().unwrap().energy = if (i == 0 || j == 0) && i != 3 && j != 3 {
                        4.0
                    } else {
                        2.0
                    };
                }
            }
        }
        let pricing = DirectOptions {
            max_arcs: 14,
            ..DirectOptions::default()
        };
        for (name, backend) in [
            (
                "native",
                EvPricingBackend::PhysicalGraph(PhysicalGraphOptions {
                    max_arcs: pricing.max_arcs,
                    ..PhysicalGraphOptions::default()
                }),
            ),
            ("direct", EvPricingBackend::Direct(pricing)),
            (
                "subpath",
                EvPricingBackend::Subpath(SubpathOptions {
                    pricing,
                    ..SubpathOptions::default()
                }),
            ),
        ] {
            let mut master = EvMaster::new(
                problem.clone(),
                ClpMaster::new()?,
                0.0,
                Some(if electric { 1.0 } else { 2.0 }),
            )?;
            let result = master.optimize(EvColumnGenerationOptions {
                backend,
                ..EvColumnGenerationOptions::default()
            })?;
            assert_eq!(
                result.status,
                EvColumnGenerationStatus::Converged,
                "{result:?}"
            );
            let objective = result.snapshot.as_ref().unwrap().travel_objective.unwrap();
            let expected = if electric { 5.0 } else { 4.5 };
            assert!((objective - expected).abs() < 1e-7);
            for column in master.columns() {
                problem.replay(&column.visits, 0.0)?;
            }
            let labels: usize = result
                .iterations
                .iter()
                .map(|i| i.pricing_statistics.labels)
                .sum();
            let subpaths: usize = result
                .iterations
                .iter()
                .map(|i| i.pricing_statistics.subpaths)
                .sum();
            let work: usize = result
                .iterations
                .iter()
                .map(|i| i.pricing_statistics.work)
                .sum();
            println!(
                "{},{name},{:?},{objective:.8},{expected:.8},{},{},{labels},{subpaths},{work}",
                if electric {
                    "charging_required"
                } else {
                    "fractional_triangle"
                },
                result.status,
                master.columns().len(),
                result.iterations.len()
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("Run with: cargo run --example evrp_compare --features clp");
}
