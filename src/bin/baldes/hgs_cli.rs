//! Opt-in BKS seeding and decomposition for the HGS command.
use crate::common_cli::{download_bks, write_checkpoint};
use std::path::Path;

use baldes::extensions::hgs::dri::{run_dri, DriOptions};
use baldes::extensions::hgs::intensify::{intensify_with_callback, IntensifyOptions, Objective};
use baldes::extensions::hgs::solution::{format_solution, parse_solution, validate_plan};
use baldes::extensions::hgs::{run_hgs, run_hgs_seeded, HgsOptions, HgsResult};
use baldes::inputs::Instance;

pub const HELP: &str = "HGS extensions (all opt-in):
  --get-bks                 Download and seed from the current CVRPLib BKS (Python 3 + internet)
  --initial-solution FILE   Seed from a local CVRPLib route file instead
  --compression-interval N  Compress consensus chains every N generations (also in subproblems)
  --delay-seed              Reserve seeds until stagnation, half time, or generated cost catches up
  --best-move              Compare candidates and apply the best move per customer
  --deterioration-credit X  First-pass credit cap in instance cost units (requires --best-move)
  --max-generations N      Total generation cap across restarts and compression stages
  --reverse-mode           Spatial route groups with progressive HGS and post-merge education
  --exploration-level N    Reverse-mode maximum depth, 0..6 (default: 4; 0 = constant)
  --decomposition-phases N Regroupings per exploration level (default: 2)
  --no-master-local-search Disable reverse-mode post-merge education for comparisons
  --intensify               Re-solve route subsets with threshold acceptance and elite restarts
  --dri                     Cluster customers, solve clusters and improve across clusters
  --dri-vicinity N          Neighboring clusters per cluster (default: 3)
  --dri-neighbors N         Similar customers considered for cross-cluster moves (default: 20)
  --dri-fuzzy X             Maximum membership of movable boundary customers (default: 0.8)
  --dri-kappa X             Fuzzy clustering exponent greater than 1 (default: 2)
  --dri-iterations N        Maximum medoid updates (default: 5)
  --dri-routing-fraction X  Fraction of remaining time for cluster routing (default: 0.8)
  --std-similarity          Rank route neighbors using spatial, temporal and demand similarity
  --barycenter             Rank route neighbors by customer-coordinate barycenters
  --workers N               Independent subproblem workers (default: 1; requires --intensify or --dri)
  --batch-size N            Jobs per solution snapshot (default: 0 = workers)
  --solution-output FILE    Checkpoint the best plan as a CVRPLib .sol file (replaces FILE)
  --intensify-rounds N      Maximum outer rounds (default: 1000)
  --subproblem-routes N     Maximum selected routes (default: 8)
  --subproblem-customers N  Maximum selected customers (default: 300)
  --subproblem-time S       Per-subproblem seconds (default: 2)
  --threshold X             Relative acceptance threshold, cooling to zero (default: 0.001)
  --elite-size N            Elite pool capacity (default: 8)
  --elite-diversity X       Minimum fraction of differing directed arcs (default: 0.05)
  --restart-after N         Restart from an elite after N stalled rounds (default: 30)
  --objective MODE          auto (default), distance, or vehicles-distance
Seeded/intensified runs use the instance metric; --get-bks uses tenths for Solomon/Homberger.
XTW uses nearest-integer distances and vehicles-first route intensification automatically.
--rounding and --objective override these defaults. Other sets default to distance.
The --time budget covers search; downloading the BKS happens before it.";

#[derive(Default)]
pub struct ExtensionOptions {
    pub get_bks: bool,
    pub initial_solution: Option<String>,
    pub intensify: bool,
    pub dri: bool,
    pub output: Option<String>,
    config: IntensifyOptions,
    objective: Option<Objective>,
    controls: bool,
    dri_vicinity: Option<usize>,
    dri_neighbors: Option<usize>,
    dri_fuzzy: Option<f64>,
    dri_kappa: Option<f64>,
    dri_iterations: Option<usize>,
    dri_routing_fraction: Option<f64>,
    route_controls: bool,
    compression_interval: Option<usize>,
    delay_seed: bool,
    best_move: bool,
    deterioration_credit: Option<f64>,
    generation_limit: Option<u32>,
    reverse_mode: bool,
    exploration_level: Option<usize>,
    decomposition_phases: Option<usize>,
    master_local_search: Option<bool>,
}

impl ExtensionOptions {
    pub fn genetic_controls_active(&self) -> bool {
        self.compression_interval.is_some()
            || self.delay_seed
            || self.best_move
            || self.deterioration_credit.is_some()
            || self.generation_limit.is_some()
            || self.reverse_mode
            || self.exploration_level.is_some()
            || self.decomposition_phases.is_some()
            || self.master_local_search.is_some()
    }

    pub fn active(&self) -> bool {
        self.get_bks
            || self.initial_solution.is_some()
            || self.intensify
            || self.dri
            || self.output.is_some()
            || self.controls
            || self.genetic_controls_active()
    }

    pub fn parse(
        &mut self,
        arg: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        match arg {
            "--get-bks" => self.get_bks = true,
            "--initial-solution" => {
                self.initial_solution = Some(args.next().ok_or("missing solution file")?)
            }
            "--compression-interval" => {
                self.compression_interval = Some(
                    args.next()
                        .ok_or("missing compression interval")?
                        .parse()
                        .map_err(|_| "invalid compression interval")?,
                );
            }
            "--delay-seed" => self.delay_seed = true,
            "--best-move" => self.best_move = true,
            "--reverse-mode" => self.reverse_mode = true,
            "--master-local-search" => self.master_local_search = Some(true),
            "--no-master-local-search" => self.master_local_search = Some(false),
            "--deterioration-credit" => {
                self.deterioration_credit = Some(
                    args.next()
                        .ok_or("missing deterioration credit")?
                        .parse()
                        .map_err(|_| "invalid deterioration credit")?,
                );
            }
            "--max-generations" => {
                self.generation_limit = Some(
                    args.next()
                        .ok_or("missing generation limit")?
                        .parse()
                        .map_err(|_| "invalid generation limit")?,
                );
            }
            "--exploration-level" => {
                self.exploration_level = Some(
                    args.next()
                        .ok_or("missing exploration level")?
                        .parse()
                        .map_err(|_| "invalid exploration level")?,
                );
            }
            "--decomposition-phases" => {
                self.decomposition_phases = Some(
                    args.next()
                        .ok_or("missing decomposition phases")?
                        .parse()
                        .map_err(|_| "invalid decomposition phases")?,
                );
            }
            "--intensify" => self.intensify = true,
            "--dri" => self.dri = true,
            "--std-similarity" => {
                self.config.std_similarity = true;
                self.controls = true;
                self.route_controls = true;
            }
            "--barycenter" => {
                self.config.barycenter = true;
                self.controls = true;
                self.route_controls = true;
            }
            "--solution-output" => {
                self.output = Some(args.next().ok_or("missing solution output file")?)
            }
            "--intensify-rounds"
            | "--workers"
            | "--batch-size"
            | "--dri-vicinity"
            | "--dri-neighbors"
            | "--dri-fuzzy"
            | "--dri-kappa"
            | "--dri-iterations"
            | "--dri-routing-fraction"
            | "--subproblem-routes"
            | "--subproblem-customers"
            | "--subproblem-time"
            | "--threshold"
            | "--elite-size"
            | "--elite-diversity"
            | "--restart-after"
            | "--objective" => {
                self.controls = true;
                if matches!(
                    arg,
                    "--batch-size"
                        | "--intensify-rounds"
                        | "--subproblem-routes"
                        | "--subproblem-time"
                        | "--threshold"
                        | "--elite-size"
                        | "--elite-diversity"
                        | "--restart-after"
                ) {
                    self.route_controls = true;
                }
                let text = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}"))?;
                let invalid = |_| format!("invalid value for {arg}: {text}");
                match arg {
                    "--intensify-rounds" => self.config.rounds = text.parse().map_err(invalid)?,
                    "--workers" => self.config.workers = text.parse().map_err(invalid)?,
                    "--batch-size" => self.config.batch_size = text.parse().map_err(invalid)?,
                    "--dri-vicinity" => self.dri_vicinity = Some(text.parse().map_err(invalid)?),
                    "--dri-neighbors" => self.dri_neighbors = Some(text.parse().map_err(invalid)?),
                    "--dri-fuzzy" => {
                        self.dri_fuzzy = Some(
                            text.parse()
                                .map_err(|_| format!("invalid value for {arg}"))?,
                        )
                    }
                    "--dri-kappa" => {
                        self.dri_kappa = Some(
                            text.parse()
                                .map_err(|_| format!("invalid value for {arg}"))?,
                        )
                    }
                    "--dri-iterations" => {
                        self.dri_iterations = Some(text.parse().map_err(invalid)?)
                    }
                    "--dri-routing-fraction" => {
                        self.dri_routing_fraction = Some(
                            text.parse()
                                .map_err(|_| format!("invalid value for {arg}"))?,
                        )
                    }
                    "--subproblem-routes" => {
                        self.config.max_routes = text.parse().map_err(invalid)?
                    }
                    "--subproblem-customers" => {
                        self.config.max_customers = text.parse().map_err(invalid)?
                    }
                    "--subproblem-time" => {
                        self.config.subproblem.time_limit = text
                            .parse()
                            .map_err(|_| format!("invalid value for {arg}"))?
                    }
                    "--threshold" => {
                        self.config.threshold = text
                            .parse()
                            .map_err(|_| format!("invalid value for {arg}"))?
                    }
                    "--elite-size" => self.config.elite_size = text.parse().map_err(invalid)?,
                    "--elite-diversity" => {
                        self.config.elite_diversity = text
                            .parse()
                            .map_err(|_| format!("invalid value for {arg}"))?
                    }
                    "--restart-after" => {
                        self.config.restart_after = text.parse().map_err(invalid)?
                    }
                    "--objective" => {
                        self.objective = match text.as_str() {
                            "auto" => None,
                            "distance" => Some(Objective::Distance),
                            "vehicles-distance" => Some(Objective::VehiclesThenDistance),
                            _ => {
                                return Err(
                                    "objective must be auto, distance or vehicles-distance".into()
                                )
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self
            .deterioration_credit
            .is_some_and(|x| !x.is_finite() || !(0. ..=f64::MAX / 10.).contains(&x))
        {
            return Err("deterioration credit must be finite and nonnegative".into());
        }
        if self.deterioration_credit.is_some_and(|x| x > 0.) && !self.best_move {
            return Err("deterioration credit requires --best-move".into());
        }
        if self.reverse_mode && (self.dri || self.intensify) {
            return Err("--reverse-mode conflicts with --intensify and --dri".into());
        }
        if self.reverse_mode && (self.config.workers != 1 || self.config.batch_size != 0) {
            return Err("--reverse-mode uses sequential route groups; --workers and --batch-size require --intensify or --dri".into());
        }
        if !self.reverse_mode
            && (self.exploration_level.is_some()
                || self.decomposition_phases.is_some()
                || self.master_local_search.is_some())
        {
            return Err(
                "exploration and master-local-search controls require --reverse-mode".into(),
            );
        }
        if self.exploration_level.is_some_and(|level| level > 6)
            || self.decomposition_phases == Some(0)
        {
            return Err(
                "exploration level must be 0..6 and decomposition phases must be positive".into(),
            );
        }
        if self.compression_interval == Some(0) {
            return Err("compression interval must be positive".into());
        }
        if self.config.barycenter && self.config.std_similarity {
            return Err("--barycenter conflicts with --std-similarity".into());
        }
        if self.get_bks && self.initial_solution.is_some() {
            return Err("--get-bks conflicts with --initial-solution".into());
        }
        if self.dri && self.intensify {
            return Err("--dri conflicts with --intensify".into());
        }
        if (self.dri || self.reverse_mode) && self.route_controls {
            return Err("route-intensification controls require --intensify".into());
        }
        if self
            .dri_fuzzy
            .is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
        {
            return Err("--dri-fuzzy must be between 0 and 1".into());
        }
        if self.dri_kappa.is_some_and(|x| !x.is_finite() || x <= 1.0) {
            return Err("--dri-kappa must be greater than 1".into());
        }
        if self
            .dri_routing_fraction
            .is_some_and(|x| !x.is_finite() || x <= 0.0 || x >= 1.0)
        {
            return Err("--dri-routing-fraction must be between 0 and 1, exclusive".into());
        }
        if self.dri_iterations == Some(0) {
            return Err("--dri-iterations must be positive".into());
        }
        if self.controls && !self.intensify && !self.dri && !self.reverse_mode {
            return Err(
                "decomposition/threshold/elite controls require --intensify or --dri".into(),
            );
        }
        if (self.dri_vicinity.is_some()
            || self.dri_neighbors.is_some()
            || self.dri_fuzzy.is_some()
            || self.dri_kappa.is_some()
            || self.dri_iterations.is_some()
            || self.dri_routing_fraction.is_some())
            && !self.dri
        {
            return Err("DRI controls require --dri".into());
        }
        if self.dri && self.objective == Some(Objective::VehiclesThenDistance) {
            return Err("--dri currently supports the distance objective only".into());
        }
        Ok(())
    }

    pub fn run(&self, instance: &Instance, options: &HgsOptions) -> Result<HgsResult, String> {
        self.validate()?;
        let mut configured = options.clone();
        if let Some(interval) = self.compression_interval {
            configured.compression = Some(baldes::extensions::hgs::CompressionOptions { interval });
        }
        configured.delay_seed |= self.delay_seed;
        configured.local_search.best_move |= self.best_move;
        if let Some(credit) = self.deterioration_credit {
            configured.local_search.max_deterioration = credit;
        }
        if let Some(limit) = self.generation_limit {
            configured.generation_limit = Some(limit);
        }
        let options = &configured;
        let objective = self.objective.unwrap_or(if instance.is_xtw() {
            Objective::VehiclesThenDistance
        } else {
            Objective::Distance
        });
        let intensify =
            self.intensify || (instance.is_xtw() && objective == Objective::VehiclesThenDistance);
        if self.dri && objective == Objective::VehiclesThenDistance {
            return Err("--dri supports distance only; use --objective distance --dri or XTW route intensification".into());
        }
        if !self.active() && !intensify {
            return run_hgs(instance, options);
        }
        let text = if self.get_bks {
            Some(download_bks(&instance.name)?)
        } else {
            self.initial_solution
                .as_ref()
                .map(std::fs::read_to_string)
                .transpose()
                .map_err(|e| e.to_string())?
        };
        let mut seed = text.as_deref().map(parse_solution).transpose()?;
        if let Some(routes) = &seed {
            validate_plan(instance, routes).map_err(|e| {
                format!("initial solution is incompatible with the instance/metric: {e}")
            })?;
        }
        let checkpoint = |routes: &[Vec<usize>], cost| {
            if let Some(path) = &self.output {
                write_checkpoint(Path::new(path), &format_solution(routes, cost))?;
            }
            Ok(())
        };
        if self.reverse_mode {
            if let Some(routes) = &seed {
                checkpoint(routes, validate_plan(instance, routes)?)?;
            }
            let config = baldes::extensions::hgs::RouteDecompositionOptions {
                time_limit: options.time_limit,
                target_customers: self.config.max_customers,
                phases: self.decomposition_phases.unwrap_or(2),
                exploration_levels: self.exploration_level.unwrap_or(4),
                master_local_search: self.master_local_search.unwrap_or(true),
                objective,
                hgs: options.clone(),
                ..Default::default()
            };
            let found =
                baldes::extensions::hgs::run_hgs_decomposed(instance, seed.as_deref(), &config)?;
            if let Some(cost) = found.solution.best_cost {
                checkpoint(&found.solution.routes, cost)?;
            }
            eprintln!("Reverse: phases={} subproblems={} levels={:?} master_passes={} master_improvements={}",
                found.decomposition.phases_completed, found.decomposition.subproblems,
                found.decomposition.exploration_schedule, found.decomposition.master_passes,
                found.decomposition.master_improvements);
            return Ok(found.solution);
        }
        if self.dri {
            if let Some(routes) = &seed {
                checkpoint(routes, validate_plan(instance, routes)?)?;
            }
            let config = DriOptions {
                time_limit: options.time_limit,
                cluster_size: self.config.max_customers,
                workers: self.config.workers,
                vicinity: self.dri_vicinity.unwrap_or(3),
                customer_neighbors: self.dri_neighbors.unwrap_or(20),
                fuzzy_threshold: self.dri_fuzzy.unwrap_or(0.8),
                fuzziness: self.dri_kappa.unwrap_or(2.0),
                clustering_iterations: self.dri_iterations.unwrap_or(5),
                routing_fraction: self.dri_routing_fraction.unwrap_or(0.8),
                hgs: options.clone(),
            };
            let result = run_dri(instance, seed.as_deref(), &config)?;
            if let Some(cost) = result.best_cost {
                checkpoint(&result.routes, cost)?;
            }
            return Ok(result);
        }
        if !intensify {
            if let Some(routes) = &seed {
                checkpoint(routes, validate_plan(instance, routes)?)?;
            }
            let result = if let Some(seed) = seed {
                run_hgs_seeded(instance, options, &[seed])?
            } else {
                run_hgs(instance, options)?
            };
            if result.best_cost.is_some() {
                checkpoint(&result.routes, validate_plan(instance, &result.routes)?)?;
            }
            return Ok(result);
        }
        let start = std::time::Instant::now();
        if seed.is_none() {
            let mut initial = options.clone();
            initial.time_limit = if options.time_limit > 0.0 {
                (options.time_limit * 0.1).min(5.0)
            } else {
                0.0
            };
            let result = run_hgs(instance, &initial)?;
            if result.best_cost.is_none() {
                return Err(
                    "no feasible initial plan; provide --get-bks or --initial-solution".into(),
                );
            }
            seed = Some(result.routes);
        }
        let seed = seed.ok_or("missing initial plan")?;
        let mut config = self.config.clone();
        config.time_limit = if options.time_limit > 0.0 {
            (options.time_limit - start.elapsed().as_secs_f64()).max(0.0)
        } else {
            0.0
        };
        config.subproblem.seed = options.seed;
        config.subproblem.nb_iter = options.nb_iter;
        config.subproblem.population_size = options.population_size;
        config.subproblem.verbose = options.verbose;
        config.subproblem.compression = options.compression.clone();
        config.subproblem.delay_seed = options.delay_seed;
        config.subproblem.local_search = options.local_search.clone();
        config.subproblem.generation_limit = options.generation_limit;
        config.objective = objective;
        if options.time_limit > 0.0 && config.time_limit == 0.0 {
            let cost = validate_plan(instance, &seed)?;
            checkpoint(&seed, cost)?;
            return Ok(HgsResult {
                routes: seed,
                columns: Vec::new(),
                best_cost: Some(cost),
                generations: 0,
                elapsed_secs: start.elapsed().as_secs_f64(),
            });
        }
        let result = intensify_with_callback(instance, &seed, &config, checkpoint)?;
        eprintln!("Intensify: initial={:.6} best={:.6} rounds={} batches={} workers={} accepted={} improvements={} restarts={} elites={}",
            result.initial_cost, result.best_cost, result.rounds, result.batches, config.workers, result.accepted, result.improvements, result.restarts, result.elite_count);
        Ok(HgsResult {
            routes: result.routes,
            columns: Vec::new(),
            best_cost: Some(result.best_cost),
            generations: result.generations,
            elapsed_secs: start.elapsed().as_secs_f64(),
        })
    }
}
