//! Equal-budget comparison with ordinary HGS; uses unrounded distances.
use baldes::extensions::hgs::{run_hgs, run_hgs_decomposed, HgsOptions, RouteDecompositionOptions};
use baldes::inputs::{Instance, Rounding};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("instances/Solomon/C101.txt");
    let seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let seeds: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    let target: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(200);
    let mut instance = Instance::read(path).unwrap();
    instance.rounding = Rounding::None;
    for seed in 0..seeds {
        let hgs = HgsOptions {
            time_limit: seconds,
            nb_iter: 300,
            seed,
            ..Default::default()
        };
        let baseline = run_hgs(&instance, &hgs).unwrap();
        println!(
            "{},policy=ordinary,seed={seed},cost={:?},seconds={:.6},routes={}",
            instance.name,
            baseline.best_cost,
            baseline.elapsed_secs,
            baseline.routes.len()
        );
        let options = RouteDecompositionOptions {
            time_limit: seconds,
            target_customers: target,
            hgs,
            ..Default::default()
        };
        match run_hgs_decomposed(&instance,None,&options) {
            Ok(result)=>println!("{},policy=decomposed,seed={seed},cost={:?},seconds={:.6},routes={},phases={},jobs={},improvements={},largest_group={}",instance.name,result.solution.best_cost,result.solution.elapsed_secs,result.solution.routes.len(),result.decomposition.phases_completed,result.decomposition.subproblems,result.decomposition.improvements,result.decomposition.largest_group),
            Err(error)=>println!("{},policy=decomposed,seed={seed},error={error}",instance.name),
        }
    }
}
