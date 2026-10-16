//! Reads a capacity separation problem from stdin and prints the cuts of each
//! round. Used by the separator comparison script.
//!
//! Input (whitespace separated): n capacity; n+1 demands; max_cuts tolerance
//! shrink candidate_limit pool two_path complement; windows flag (then n+1
//! ready, due, service and (n+1)^2 travel); rounds; per round the column
//! count, then per column its value, length and customers.
use baldes::cuts::capacity::{CapacitySeparator, TimeWindows};
use std::io::Read;

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut it = text.split_whitespace();
    let mut next = || it.next().unwrap().to_string();
    let n: usize = next().parse().unwrap();
    let capacity: f64 = next().parse().unwrap();
    let demands: Vec<f64> = (0..=n).map(|_| next().parse().unwrap()).collect();
    let mut separator = CapacitySeparator::new(demands, capacity).unwrap();
    let p = &mut separator.parameters;
    p.max_cuts_per_round = next().parse().unwrap();
    p.violation_tolerance = next().parse().unwrap();
    p.shrink_threshold = next().parse().unwrap();
    p.candidate_limit = next().parse().unwrap();
    p.use_global_pool = next() == "1";
    p.use_two_path = next() == "1";
    p.complement_form = next() == "1";
    if next() == "1" {
        let mut vec = |k: usize| -> Vec<f64> { (0..k).map(|_| next().parse().unwrap()).collect() };
        let ready = vec(n + 1);
        let due = vec(n + 1);
        let service = vec(n + 1);
        let travel = (0..=n).map(|_| vec(n + 1)).collect();
        separator.set_time_windows(TimeWindows { ready, due, service, travel }).unwrap();
    }
    let rounds: usize = next().parse().unwrap();
    for round in 0..rounds {
        let count: usize = next().parse().unwrap();
        let mut columns = vec![];
        for _ in 0..count {
            let value: f64 = next().parse().unwrap();
            let len: usize = next().parse().unwrap();
            let customers: Vec<usize> = (0..len).map(|_| next().parse().unwrap()).collect();
            columns.push((customers, value));
        }
        let cuts = separator.separate(&columns).unwrap();
        println!("round {round} {}", cuts.len());
        for c in cuts {
            let kind = match c.kind {
                baldes::cuts::capacity::CapacityKind::Capacity => 0,
                baldes::cuts::capacity::CapacityKind::TwoPath => 1,
                baldes::cuts::capacity::CapacityKind::Complement => 2,
            };
            let customers: Vec<String> = c.customers.iter().map(|v| v.to_string()).collect();
            let edges: Vec<String> = c.edges.iter().map(|((a, b), w)| format!("{a}-{b}:{w}")).collect();
            println!("cut {kind} {} {:.17e} {} | {}", c.rhs, c.violation, customers.join(","), edges.join(","));
        }
    }
}
