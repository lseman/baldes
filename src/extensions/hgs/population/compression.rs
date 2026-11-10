use super::*;
use crate::extensions::hgs::compression::{model::Plan, Consensus};

type Routes = Vec<Vec<usize>>;

#[derive(Clone)]
struct Saved {
    routes: Routes,
    distance: f64,
    capacity: i32,
    warp: f64,
    feasible: bool,
}
impl Saved {
    fn capture(i: &Individual<'_>) -> Self {
        Self {
            routes: i
                .chrom_r
                .iter()
                .filter(|r| !r.is_empty())
                .cloned()
                .collect(),
            distance: i.my_cost_sol.distance,
            capacity: i.my_cost_sol.capacity_excess,
            warp: i.my_cost_sol.time_warp,
            feasible: i.is_feasible,
        }
    }
    fn individual<'p>(&self, params: &'p Params) -> Individual<'p> {
        let mut i = Individual::new_empty(params);
        for (target, route) in i.chrom_r.iter_mut().zip(&self.routes) {
            target.clone_from(route);
        }
        i.chrom_t = self.routes.iter().flatten().copied().collect();
        i.evaluate_complete_cost();
        i
    }
    fn project(&self, params: &Params, plan: &Plan) -> Option<Self> {
        let routes = plan.project(&self.routes)?;
        if routes.len() > params.nb_vehicles as usize {
            return None;
        }
        let mut saved = self.clone();
        saved.routes = routes;
        let i = saved.individual(params);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.);
        if !close(self.distance, i.my_cost_sol.distance + plan.frozen_distance)
            || self.capacity != i.my_cost_sol.capacity_excess
            || !close(self.warp, i.my_cost_sol.time_warp)
            || self.feasible != i.is_feasible
            || self.routes.len() != i.my_cost_sol.nb_routes + plan.fixed.len()
        {
            return None;
        }
        Some(Self::capture(&i))
    }
}

pub(crate) struct Checkpoint {
    reserved_seeds: Vec<Saved>,
    feasible: Vec<Saved>,
    infeasible: Vec<Saved>,
    best_restart: Option<Saved>,
    best_overall: Option<Saved>,
    elite: Vec<Saved>,
    load_window: VecDeque<bool>,
    tw_window: VecDeque<bool>,
    progress: Vec<(f64, f64)>,
    elite_max_stale: usize,
    elite_stale: usize,
    generations: u32,
    minimum: i32,
    objective_offset: f64,
    frozen_routes: usize,
}
impl Checkpoint {
    pub(crate) fn project(mut self, params: &Params, plan: &Plan) -> Option<Self> {
        self.objective_offset += plan.frozen_distance;
        self.frozen_routes += plan.fixed.len();
        let project_all = |items: Vec<Saved>| {
            items
                .iter()
                .map(|s| s.project(params, plan))
                .collect::<Option<Vec<_>>>()
        };
        self.reserved_seeds = project_all(self.reserved_seeds)?;
        self.feasible = project_all(self.feasible)?;
        self.infeasible = project_all(self.infeasible)?;
        self.elite = project_all(self.elite)?;
        if let Some(saved) = self.best_restart.take() {
            self.best_restart = Some(saved.project(params, plan)?);
        }
        if let Some(saved) = self.best_overall.take() {
            self.best_overall = Some(saved.project(params, plan)?);
        }
        Some(self)
    }
    pub(crate) fn restore<'p>(self, params: &'p Params) -> Population<'p> {
        let mut population = Population::new(params);
        population.objective_offset = self.objective_offset;
        population.frozen_routes = self.frozen_routes;
        population.minimum_population_size = self.minimum;
        for saved in self
            .feasible
            .iter()
            .rev()
            .chain(self.infeasible.iter().rev())
        {
            population.add_individual(&saved.individual(params), false);
        }
        if let Some(best) = self.best_restart {
            population.best_restart = Box::new(best.individual(params));
        }
        if let Some(best) = self.best_overall {
            population.best_overall = Box::new(best.individual(params));
        }
        population.mdm_elite = self
            .elite
            .iter()
            .map(|s| Box::new(s.individual(params)))
            .collect();
        population.reserved_seeds = self
            .reserved_seeds
            .iter()
            .map(|s| s.individual(params))
            .collect();
        population.mdm_elite_updated = true;
        population.mdm_patterns.clear();
        population.last_feas_load = self.load_window;
        population.last_feas_tw = self.tw_window;
        population.search_progress = self.progress;
        population.mdm_elite_max_non_updating_restarts = self.elite_max_stale;
        population.mdm_elite_non_updating_restarts = self.elite_stale;
        population.nb_generations = self.generations;
        population
    }
}

impl Population<'_> {
    pub(crate) fn compression_plan(&mut self) -> Option<Plan> {
        let mut consensus = self.consensus.take()?;
        self.consensus = Some(Consensus::new(self.params.nb_clients as usize));
        // A feasible epoch observation is mandatory. Screen survivors (including
        // infeasible ones), incumbents and elites; none may be silently dropped.
        if !consensus.started() {
            return None;
        }
        for i in self
            .feasible
            .iter()
            .chain(&self.infeasible)
            .chain(&self.mdm_elite)
            .chain([&self.best_restart, &self.best_overall])
        {
            if i.my_cost_sol.penalized_cost < 1e29 {
                consensus.observe(i);
            }
        }
        for seed in &self.reserved_seeds {
            consensus.observe(seed);
        }
        consensus.plan(self.params)
    }
    pub(crate) fn checkpoint(&self) -> Checkpoint {
        let saved =
            |i: &Individual<'_>| (i.my_cost_sol.penalized_cost < 1e29).then(|| Saved::capture(i));
        Checkpoint {
            reserved_seeds: self.reserved_seeds.iter().map(Saved::capture).collect(),
            feasible: self.feasible.iter().map(|i| Saved::capture(i)).collect(),
            infeasible: self.infeasible.iter().map(|i| Saved::capture(i)).collect(),
            best_restart: saved(&self.best_restart),
            best_overall: saved(&self.best_overall),
            elite: self.mdm_elite.iter().map(|i| Saved::capture(i)).collect(),
            load_window: self.last_feas_load.clone(),
            tw_window: self.last_feas_tw.clone(),
            progress: self.search_progress.clone(),
            elite_max_stale: self.mdm_elite_max_non_updating_restarts,
            elite_stale: self.mdm_elite_non_updating_restarts,
            generations: self.nb_generations,
            minimum: self.minimum_population_size,
            objective_offset: self.objective_offset,
            frozen_routes: self.frozen_routes,
        }
    }
}
