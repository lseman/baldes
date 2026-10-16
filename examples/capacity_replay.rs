//! Replay capacity-separator snapshots: n Q rounds, n demands, then per-round
//! column count followed by lines `value length customer...` (depot implicit).
use baldes::cuts::capacity::CapacitySeparator;
fn main() {
    let input = std::fs::read_to_string(std::env::args().nth(1).expect("input file")).unwrap();
    let mut tokens = input.split_whitespace();
    let mut number = || tokens.next().expect("input token").parse::<f64>().unwrap();
    let n = number() as usize;
    let capacity = number();
    let rounds = number() as usize;
    let mut demands = vec![0.];
    for _ in 0..n {
        demands.push(number());
    }
    let mut sep = CapacitySeparator::new(demands, capacity).unwrap();
    sep.parameters.complement_form = true;
    sep.parameters.use_two_path = false;
    for round in 1..=rounds {
        let count = number() as usize;
        let mut columns = vec![];
        for _ in 0..count {
            let value = number();
            let len = number() as usize;
            let nodes = (0..len).map(|_| number() as usize).collect();
            columns.push((nodes, value));
        }
        let cuts = sep.separate(&columns).unwrap();
        let records: Vec<_> = cuts
            .iter()
            .map(|c| {
                let edges: Vec<_> = c
                    .edges
                    .iter()
                    .map(|(&(a, b), v)| format!("[{a},{b},{v}]"))
                    .collect();
                format!(
                    "{{\"customers\":{:?},\"rhs\":{},\"type\":{},\"edges\":[{}]}}",
                    c.customers,
                    c.rhs,
                    if c.greater_equal() { 0 } else { 1 },
                    edges.join(",")
                )
            })
            .collect();
        println!("{{\"call\":{round},\"cuts\":[{}]}}", records.join(","));
    }
}
