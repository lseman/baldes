//! One command frontend for routing and collections of pricing networks.
use super::{ce_cli, pstep_cli};

mod networks;
mod routing;

use baldes::inputs::{Instance, Rounding};
use baldes::master::model::{
    self,
    format::{Model, ResourceGroupRestriction, SolveOptions, SolveResult},
};
use baldes::master::routing::RootOptions;
use networks::{solve_networks, NETWORK_HELP};
use routing::{solve_routing, ROUTING_HELP};

pub(super) fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(ROUTING_HELP)?;
    match command.as_str() {
        "-h" | "--help" => {
            println!(
                "{ROUTING_HELP}\n\n{}\n\n{NETWORK_HELP}",
                super::hgs_cli::HELP
            );
            Ok(())
        }
        "pstep" => pstep_cli::run(args.collect()),
        "ce" => ce_cli::run(args.collect()),
        "solve" | "hgs" => {
            let file = args.next().ok_or(ROUTING_HELP)?;
            if file == "--help" || file == "-h" {
                println!(
                    "{ROUTING_HELP}\n\n{}\n\n{NETWORK_HELP}",
                    super::hgs_cli::HELP
                );
                return Ok(());
            }
            if file.starts_with('-') {
                return Err(format!("expected model or instance path, got {file}"));
            }
            if command == "solve"
                && std::path::Path::new(&file)
                    .extension()
                    .is_some_and(|e| e == "json")
            {
                solve_networks(file, args.collect())
            } else {
                solve_routing(command, file, args.collect())
            }
        }
        _ => Err(format!("unknown command: {command}\n{ROUTING_HELP}")),
    }
}

/// The same result sink is used for routing and network models.
fn write_json(path: &str, data: &str) -> Result<bool, String> {
    if path == "-" {
        println!("{}", data.trim_end());
        Ok(true)
    } else {
        std::fs::write(path, data).map_err(|e| e.to_string())?;
        Ok(false)
    }
}

#[derive(Default)]
struct CommonOptions {
    time: Option<std::time::Duration>,
    iterations: Option<usize>,
    nodes: Option<usize>,
    bound_mode: Option<baldes::master::cg::BoundMode>,
    stabilize: Option<bool>,
    depth: Option<i64>,
    log_interval: Option<usize>,
    pricing: Vec<(String, f64)>,
    tree: bool,
    json: Option<String>,
}
impl CommonOptions {
    fn parse(&mut self, arg: &str, args: &mut std::vec::IntoIter<String>) -> Result<bool, String> {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing {}", arg.trim_start_matches('-').replace('-', " ")))
        };
        match arg {
            "--tree" => self.tree = true,
            "--json" => self.json = Some(value()?),
            "--no-stabilization" => self.stabilize = Some(false),
            "--verbose" | "--log" | "-v" => {
                self.log_interval.get_or_insert(10);
            }
            "--time" => {
                self.time = Some(
                    std::time::Duration::try_from_secs_f64(
                        value()?.parse::<f64>().map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| e.to_string())?,
                )
            }
            "--iterations" => {
                self.iterations = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--nodes" => self.nodes = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?),
            "--stabilization-max-tree-depth" => {
                self.depth = Some(value()?.parse::<i64>().map_err(|e| e.to_string())?)
            }
            "--bound-mode" => {
                self.bound_mode = Some(
                    baldes::master::cg::BoundMode::parse(&value()?).map_err(|e| e.to_string())?,
                )
            }
            "--log-interval" => {
                let n = value()?.parse::<usize>().map_err(|e| e.to_string())?;
                if n == 0 {
                    return Err("log interval must be positive".into());
                }
                self.log_interval = Some(n);
            }
            "--option" => {
                let assignment = value()?;
                let (name, v) = assignment
                    .split_once('=')
                    .ok_or("option must be NAME=VALUE")?;
                self.pricing
                    .push((name.into(), v.parse::<f64>().map_err(|e| e.to_string())?));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
    fn apply_routing(&self, options: &mut RootOptions) {
        if let Some(v) = self.time {
            options.cg.time_limit = v;
        }
        if let Some(v) = self.iterations {
            options.cg.max_iterations = v;
        }
        if let Some(v) = self.bound_mode {
            options.bound_mode = v;
        }
        if let Some(v) = self.stabilize {
            options.cg.stabilize = v;
        }
        if let Some(v) = self.depth {
            options.cg.stabilization_max_tree_depth = v;
        }
        if let Some(v) = self.log_interval {
            options.cg.log_interval = Some(v);
        }
        options.pricing_options.extend(self.pricing.iter().cloned());
    }
    fn apply_networks(&self, options: &mut SolveOptions) {
        options.tree = self.tree;
        if let Some(v) = self.nodes {
            options.node_limit = v;
        }
        if let Some(v) = self.time {
            options.time_limit = v.as_secs_f64();
        }
        if let Some(v) = self.iterations {
            options.max_iterations = v;
        }
        if let Some(v) = self.bound_mode {
            options.bound_mode = v.as_str().into();
        }
        if let Some(v) = self.stabilize {
            options.stabilize = v;
        }
        if let Some(v) = self.depth {
            options.stabilization_max_tree_depth = v;
        }
        if let Some(v) = self.log_interval {
            options.log_interval = Some(v);
        }
        options.pricing_options.extend(self.pricing.iter().cloned());
    }
}
