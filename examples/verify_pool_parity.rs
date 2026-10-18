//! Text-protocol driver for grouped pricing-candidate binary comparisons.
use baldes::labeling::pool::SolutionPool;
use std::io::{self, Read};
fn read<T: std::str::FromStr>(input: &mut std::str::SplitWhitespace<'_>) -> T
where
    T::Err: std::fmt::Debug,
{
    input.next().unwrap().parse().unwrap()
}
fn main() {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text).unwrap();
    let mut input = text.split_whitespace();
    let cases: usize = read(&mut input);
    for case in 0..cases {
        let groups = read(&mut input);
        let total = read(&mut input);
        let limit = read(&mut input);
        let threshold = read(&mut input);
        let steps: usize = read(&mut input);
        let mut pool = SolutionPool::new(groups, total, limit);
        pool.threshold = threshold;
        for step in 0..steps {
            let group = read(&mut input);
            let cost = read(&mut input);
            let secondary = read(&mut input);
            pool.insert(group, cost, secondary, step);
            // Accepted candidates consume exactly one monotonically increasing ID.
            println!(
                "POOL {case} {step} {} {} {} {}",
                pool.inserted,
                pool.inserted,
                pool.threshold,
                pool.len()
            );
            for c in pool.iter() {
                println!(
                    "CAND {} {} {} {} {}",
                    c.id, c.group, c.cost, c.secondary, c.value
                );
            }
        }
    }
    assert!(input.next().is_none());
}
