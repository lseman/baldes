//! Compare ordinary HGS and consensus compression at equal time budgets.
//! cargo run --release --example hgs_compression_bench -- instances/Solomon/C101.txt 2 3 75
use baldes::extensions::hgs::{run_hgs, run_hgs_compressed, CompressionOptions, HgsOptions};
use baldes::inputs::{Instance, Rounding};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("instances/Solomon/C101.txt");
    let seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2.);
    let seeds: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    let interval: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(75);
    let mut instance = Instance::read(path).unwrap();
    instance.rounding = Rounding::None;
    for seed in 0..seeds {
        let options = HgsOptions {
            time_limit: seconds,
            nb_iter: 300,
            seed,
            ..Default::default()
        };
        let baseline = run_hgs(&instance, &options).unwrap();
        println!(
            "{},policy=ordinary,seed={seed},generations={},cost={:?},seconds={:.6}",
            instance.name, baseline.generations, baseline.best_cost, baseline.elapsed_secs
        );
        let result =
            run_hgs_compressed(&instance, &options, &CompressionOptions { interval }).unwrap();
        println!("{},policy=compressed,seed={seed},generations={},cost={:?},seconds={:.6},checks={},contractions={},active_customers={},frozen_routes={}",
            instance.name,result.solution.generations,result.solution.best_cost,result.solution.elapsed_secs,
            result.compression.checks,result.compression.contractions,result.compression.final_customers,result.compression.frozen_routes);
    }
}
