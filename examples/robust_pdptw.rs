//! cargo run --release --features clp --example robust_pdptw -- [FILE Γq Γt σq σt SECONDS ROUNDING NG]
#[cfg(feature = "clp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use baldes::extensions::robust_pd::{DistanceRounding, Instance, Parameters, Problem};
    use baldes::master::tree::ModelTreeOptions;
    use std::time::{Duration, Instant};

    fn invalid(message: &str) -> baldes::core::error::OrError {
        baldes::core::error::OrError {
            kind: baldes::core::error::ErrorKind::InvalidArgument,
            message: message.to_owned(),
        }
    }
    fn solve() -> baldes::Result<()> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            println!("robust_pdptw [FILE Γq Γt σq σt SECONDS ROUNDING NG]\nDefaults: instances/RobustPDPTW/AA05.txt 1 1 0.1 0.1 60 hundredth 8\nROUNDING: hundredth or none. NG: requests per NG neighbourhood; at least the request count prices elementary routes. Paper §6.1 demand adjustment enabled.");
            return Ok(());
        }
        if args.len() > 8 {
            return Err(invalid("expected FILE Γq Γt σq σt SECONDS ROUNDING NG"));
        }
        let number = |i: usize, default: &str| args.get(i).map_or(default.to_owned(), Clone::clone);
        let parse_error = || invalid("invalid numeric argument");
        let parameters = Parameters {
            demand_budget: number(1, "1").parse().map_err(|_| parse_error())?,
            time_budget: number(2, "1").parse().map_err(|_| parse_error())?,
            demand_deviation: number(3, "0.1").parse().map_err(|_| parse_error())?,
            time_deviation: number(4, "0.1").parse().map_err(|_| parse_error())?,
            distance_rounding: match number(6, "hundredth").as_str() {
                "hundredth" => DistanceRounding::NearestHundredth,
                "none" => DistanceRounding::FullPrecision,
                _ => return Err(invalid("rounding must be hundredth or none")),
            },
            ng_size: number(7, "8").parse().map_err(|_| parse_error())?,
            ..Default::default()
        };
        let seconds: f64 = number(5, "60").parse().map_err(|_| parse_error())?;
        if !seconds.is_finite() || !(0.001..=86400.).contains(&seconds) {
            return Err(invalid("seconds must be in [0.001,86400]"));
        }
        let path = args
            .first()
            .map_or("instances/RobustPDPTW/AA05.txt", String::as_str);
        let problem = Problem::new(Instance::read(path)?, parameters)?;
        let mut options = ModelTreeOptions {
            time_limit: Duration::from_secs_f64(seconds),
            ..Default::default()
        };
        options.node.cg.time_limit = Duration::from_secs_f64(seconds);
        options.node.cg.artificial_cost =
            (parameters.vehicle_cost + 100_000.) * problem.instance().requests() as f64;
        let started = Instant::now();
        let output = baldes::master::tree::solve(problem.master_problem()?, options)?;
        println!(
            "instance={path} gamma_q={} gamma_t={} sigma_q={} sigma_t={} ng={}",
            parameters.demand_budget,
            parameters.time_budget,
            parameters.demand_deviation,
            parameters.time_deviation,
            parameters.ng_size
        );
        println!(
            "status={:?} bound={:.6} incumbent={:.6} nodes={} pricing_calls={} seconds={:.3}",
            output.status,
            output.bound,
            output.incumbent,
            output.nodes,
            output.pricing_calls,
            started.elapsed().as_secs_f64()
        );
        let mut coverage = vec![0.; problem.instance().requests()];
        let mut cost = 0.;
        for fixed in &output.routes {
            let route: Vec<usize> = fixed
                .route
                .route
                .nodes
                .iter()
                .map(|v| *v as usize)
                .collect();
            let schedule = problem.replay(&route)?;
            for &v in &route {
                if (1..=coverage.len()).contains(&v) {
                    coverage[v - 1] += fixed.quantity;
                }
            }
            cost += fixed.quantity * schedule.cost;
            println!(
                "quantity={} route={route:?} cost={:.6} worst_load={:.3} worst_return={:.3}",
                fixed.quantity,
                schedule.cost,
                schedule.loads.iter().copied().fold(0., f64::max),
                schedule.starts.last().unwrap().last().unwrap()
            );
        }
        if output.incumbent.is_finite()
            && (coverage.iter().any(|x| (*x - 1.).abs() > 1e-6)
                || (cost - output.incumbent).abs() > 1e-5)
        {
            return Err(invalid("incumbent replay/coverage mismatch"));
        }
        Ok(())
    }
    solve().map_err(|e| e.to_string().into())
}

#[cfg(not(feature = "clp"))]
fn main() {
    eprintln!("robust_pdptw requires --features clp");
    std::process::exit(1);
}
