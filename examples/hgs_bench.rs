//! Fixed-work HGS benchmark: cargo run --release --example hgs_bench -- Solomon/R201 300 3
use baldes::extensions::hgs::driver::{run_hgs, HgsOptions};
use baldes::inputs::Instance;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).map(String::as_str).unwrap_or("Solomon/R201");
    let iterations = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(300);
    let repeats = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    let instance = Instance::read(format!(
        "{}/instances/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    for seed in 0..repeats {
        let options = HgsOptions {
            time_limit: 0.0,
            nb_iter: iterations,
            seed,
            ..Default::default()
        };
        let result = run_hgs(&instance, &options).unwrap();
        println!(
            "{name},seed={seed},generations={},cost={:?},seconds={:.6}",
            result.generations, result.best_cost, result.elapsed_secs
        );
    }
}
