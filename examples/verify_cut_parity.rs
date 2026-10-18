//! Text-protocol driver for tools/cut_parity/compare.py.
use baldes::cuts::capacity::{CapacityKind, CapacitySeparator};
use std::io::{self, Read};
fn main() {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text).unwrap();
    let mut input = text.split_whitespace();
    fn read<T: std::str::FromStr>(input: &mut std::str::SplitWhitespace<'_>) -> T
    where
        T::Err: std::fmt::Debug,
    {
        input.next().unwrap().parse().unwrap()
    }
    let cases: usize = read(&mut input);
    for case in 0..cases {
        let count: usize = read(&mut input);
        let capacity: f64 = read(&mut input);
        let complement: i32 = read(&mut input);
        let rounds: usize = read(&mut input);
        let mut demands = vec![0.];
        for _ in 0..count {
            demands.push(read(&mut input));
        }
        let mut sep = CapacitySeparator::new(demands, capacity).unwrap();
        sep.parameters.complement_form = complement != 0;
        sep.parameters.use_two_path = false;
        for round in 0..rounds {
            let columns: usize = read(&mut input);
            let mut routes = vec![];
            for _ in 0..columns {
                let value: f64 = read(&mut input);
                let len: usize = read(&mut input);
                let path = (0..len).map(|_| read(&mut input)).collect::<Vec<usize>>();
                routes.push((path, value));
            }
            let cuts = sep.separate(&routes).unwrap();
            let stats = sep.statistics();
            println!(
                "ROUND {case} {round} {} {} {} {} {}",
                stats.calls,
                stats.cuts_created,
                stats.columns,
                stats.global_subsets,
                cuts.len()
            );
            for cut in cuts {
                let kind = if cut.kind == CapacityKind::Complement {
                    1
                } else {
                    0
                };
                print!("CUT {} {kind} {} {}", cut.id, cut.rhs, cut.customers.len());
                for v in cut.customers {
                    print!(" {v}");
                }
                print!(" {}", cut.edges.len());
                for ((u, v), w) in cut.edges {
                    print!(" {u} {v} {w}");
                }
                println!();
            }
        }
    }
    assert!(input.next().is_none());
}
