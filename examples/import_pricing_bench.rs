//! Imported pricing-query microbenchmark; setup and LP solving are untimed.
//! cargo run --release --features clp --example import_pricing_bench -- 256 2000
use baldes::master::{
    cg::BoundMode,
    mp::{
        imported::{self, Column, ColumnEnvelope, Subproblem},
        MasterVariable, Options, Problem,
    },
    preprocessing::Bounds,
};
use std::{hint::black_box, time::Instant};

fn main() -> baldes::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let size: usize = args
        .get(1)
        .map_or(256, |s| s.parse().expect("column count"));
    let queries: usize = args
        .get(2)
        .map_or(2000, |s| s.parse().expect("query count"));
    assert!(size > 0 && size < i32::MAX as usize && queries > 0);
    let bounds = |lower, upper| Bounds { lower, upper };
    let variables = (0..8)
        .map(|id| MasterVariable {
            id,
            cost: 0.,
            bounds: bounds(0., 1000.),
            integer: false,
            entries: vec![],
        })
        .collect();
    let groups = (0..4)
        .map(|id| Subproblem {
            id,
            variables: (0..8).collect(),
            multiplicity: bounds(0., 100.),
            columns: (0..size)
                .map(|c| Column {
                    id: c as i32,
                    cost: 1. + c as f64 / 1024.,
                    bounds: bounds(0., f64::INFINITY),
                    integer: false,
                    entries: vec![(0, 1.), (3, 0.5), (5, 0.25), (7, 0.125)],
                })
                .collect(),
        })
        .collect();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = BoundMode::Numerical;
    let root = imported::solve(
        Problem::<2> {
            rows: vec![],
            networks: vec![],
        },
        variables,
        groups,
        options.clone(),
    )?;
    let session = root
        .snapshot()?
        .begin_imports(&[0, 1, 2, 3])?
        .with_envelope(
            0,
            ColumnEnvelope {
                cost_lower: 0.,
                coordinates: (0..8).map(|id| (id, bounds(0., 1.))).collect(),
            },
        )?;
    let point = session.pricing(0, options)?;
    let candidate = Column {
        id: size as i32,
        cost: 0.5,
        bounds: bounds(0., f64::INFINITY),
        integer: false,
        entries: vec![(7, 0.125), (0, 1.), (5, 0.25), (3, 0.5)],
    };
    println!("columns_per_group={size}, groups=4, queries={queries}");
    // Repeat all query types so baseline/candidate process pairs can compare
    // medians. black_box retains both the inputs and returned allocations.
    for sample in 0..5 {
        let start = Instant::now();
        for _ in 0..queries {
            black_box(point.reduced_cost(black_box(candidate.clone()))?);
        }
        let reduced = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        for _ in 0..queries {
            black_box(point.costs()?);
        }
        let costs = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        for _ in 0..queries {
            black_box(point.envelope_reduced_cost_bound()?);
        }
        let envelope = start.elapsed().as_secs_f64() * 1000.;
        println!(
            "sample={sample} reduced_ms={reduced:.6} costs_ms={costs:.6} envelope_ms={envelope:.6}"
        );
    }
    Ok(())
}
