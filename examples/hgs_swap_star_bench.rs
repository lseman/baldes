//! Compare broad and filtered SWAP* at equal time budgets and seeds.
//! cargo run --release --example hgs_swap_star_bench -- instances/Solomon/R201.txt 2 3
//! Set seconds to zero for a fixed 300-stagnation-iteration comparison.
use baldes::extensions::hgs::{
    genetic::Genetic, local_search::HGSLocalSearch, Params, Population, Split,
};
use baldes::inputs::{Instance, Rounding};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("instances/Solomon/R201.txt");
    let seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2.);
    let seeds: i32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);
    assert!(seconds.is_finite() && seconds >= 0. && seeds > 0);
    let mut instance = Instance::read(path).unwrap();
    instance.rounding = Rounding::None;
    for seed in 0..seeds {
        for (policy, neighbors) in [("broad", 0), ("filtered", 35)] {
            let mut params = Params::from_instance(&instance);
            params.config.time_limit = seconds;
            params.config.nb_iter = 300;
            params.config.swap_star_neighbors = neighbors;
            params.config.swap_star_demand_percentage = 30;
            params.config.swap_star_full_pass_interval = 5;
            params.verbose = false;
            params.set_seed(seed);
            params.reset_clock();
            let mut split = Split::new(&params);
            let mut local_search = HGSLocalSearch::new(&params);
            let mut population = Population::new(&params);
            population.generate_population(&mut split, &mut local_search);
            Genetic::new(&params, &mut split, &mut local_search).run(&mut population, 300);
            let best = population.get_best_found().filter(|i| i.is_feasible);
            println!(
                "{},policy={policy},seed={seed},generations={},cost={:?},seconds={:.6}",
                instance.name,
                population.nb_generations,
                best.map(|i| i.my_cost_sol.distance / 10.),
                params.elapsed_secs()
            );
        }
    }
}
