//! Bounded background adapter for the shared AILS-II engine.
use super::*;
use crate::{
    extensions::{
        ails_ii::{run_ails_ii_controlled, AilsIiOptions},
        common::route_subproblem,
    },
    inputs::{HeuristicRouteMap, Instance},
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    },
    thread::JoinHandle,
    time::Duration,
};

/// Run the existing AILS-II heuristic on disjoint feasible routes from one
/// nominal CVRP/VRPTW network. Instance IDs must match its split-depot graph.
#[derive(Clone, Debug)]
pub struct BackgroundIlsOptions {
    pub instance: Arc<Instance>,
    pub network: i32,
    /// Maximum seed routes in one customer subproblem.
    pub max_routes: usize,
    /// Controls of the same engine used by the standalone `ails-ii` command.
    /// Background jobs require a positive time limit.
    pub search: AilsIiOptions,
}

impl BackgroundIlsOptions {
    pub fn new(instance: Instance) -> Self {
        Self {
            instance: Arc::new(instance),
            network: 0,
            max_routes: 16,
            search: AilsIiOptions {
                time_limit: 0.1,
                iterations: 20,
                seed: 1,
                ..Default::default()
            },
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.search.validate().map_err(invalid)?;
        crate::extensions::common::validate_instance(&self.instance).map_err(invalid)?;
        if self.max_routes == 0 || self.search.time_limit <= 0. || self.instance.customers.len() < 2
        {
            return Err(invalid(
                "background AILS-II requires an instance, seed routes and a positive time budget",
            ));
        }
        Duration::try_from_secs_f64(self.search.time_limit)
            .map_err(|_| invalid("background AILS-II time budget"))?;
        Ok(())
    }
}

struct Job<const D: usize> {
    graph: Arc<Graph<D>>,
    seeds: Vec<(f64, Arc<NetworkRoute>)>,
    deadline: Instant,
}

pub(super) struct Worker<const D: usize> {
    network: i32,
    customers: usize,
    cancel: Arc<AtomicBool>,
    requests: Option<SyncSender<Job<D>>>,
    results: Receiver<Vec<NetworkRoute>>,
    thread: Option<JoinHandle<()>>,
    busy: bool,
}

impl<const D: usize> Worker<D> {
    pub(super) fn new(options: BackgroundIlsOptions) -> Result<Self> {
        options.validate()?;
        let network = options.network;
        let customers = options.instance.customers.len();
        let cancel = Arc::new(AtomicBool::new(false));
        let (requests, jobs) = mpsc::sync_channel::<Job<D>>(1);
        let (completed, results) = mpsc::sync_channel(1);
        let stop = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("baldes-ails-ii".into())
            .spawn(move || {
                let mut cached: Option<(Arc<Graph<D>>, HeuristicRouteMap)> = None;
                let mut sequence = 0_i32;
                while let Ok(job) = jobs.recv() {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if cached
                        .as_ref()
                        .is_none_or(|(graph, _)| !Arc::ptr_eq(graph, &job.graph))
                    {
                        cached = Some((job.graph.clone(), HeuristicRouteMap::new(&job.graph)));
                    }
                    let seed = options.search.seed.wrapping_add(sequence);
                    sequence = sequence.wrapping_add(1);
                    let routes =
                        search_job(&options, job, &cached.as_ref().unwrap().1, &stop, seed);
                    // One outstanding request and a single result slot; never block shutdown.
                    if completed.try_send(routes).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| invalid(format!("start background AILS-II: {e}")))?;
        Ok(Self {
            network,
            customers,
            cancel,
            requests: Some(requests),
            results,
            thread: Some(thread),
            busy: false,
        })
    }

    pub(super) fn poll(&mut self) -> Result<Vec<NetworkRoute>> {
        match self.results.try_recv() {
            Ok(routes) => {
                self.busy = false;
                Ok(routes)
            }
            Err(TryRecvError::Empty) => Ok(vec![]),
            Err(TryRecvError::Disconnected) => {
                Err(invalid("background AILS-II worker stopped unexpectedly"))
            }
        }
    }

    pub(super) fn submit(&mut self, state: &State<D>, columns: &[cg::Column]) -> Result<()> {
        if self.busy
            || state.universe.is_some()
            || state.gap_pool_cutoff.is_some()
            || Instant::now() >= state.deadline
        {
            return Ok(());
        }
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
            .ok_or_else(|| invalid("background AILS-II network is absent"))?;
        if oracle.solver.enumeration.is_some() || oracle.multiplicity_upper() == 0. {
            return Ok(());
        }
        let graph = &oracle.network.graph;
        if graph.has_coupled_resources()
            || graph.source != 0
            || graph.sink != self.customers
            || graph.vertices.len() != self.customers + 1
        {
            return Err(invalid(
                "background AILS-II requires its instance's nominal split-depot graph",
            ));
        }
        let seeds: Vec<_> = columns
            .iter()
            .filter_map(|column| {
                let route = state.routes.get(usize::try_from(column.tag).ok()?)?;
                (route.network == self.network && route.route.nodes.len() >= 3)
                    .then(|| (column.reduced_cost, route.clone()))
            })
            .collect();
        if seeds.is_empty() {
            return Ok(());
        }
        match self.requests.as_ref().unwrap().try_send(Job {
            graph: graph.clone(),
            seeds,
            deadline: state.deadline,
        }) {
            Ok(()) => {
                self.busy = true;
                Ok(())
            }
            Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                Err(invalid("background AILS-II worker stopped unexpectedly"))
            }
        }
    }
}

impl<const D: usize> Drop for Worker<D> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.requests.take(); // Wake an idle recv(). Active searches observe their deadline.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn stopped(cancel: &AtomicBool, deadline: Instant) -> bool {
    // Cancellation carries no data.
    cancel.load(Ordering::Relaxed) || Instant::now() >= deadline
}

/// Select a disjoint route neighborhood; its union is a complete seed for the
/// customer subproblem. All ranking, replay and customer copying run off-thread.
fn neighborhood<const D: usize>(
    options: &BackgroundIlsOptions,
    job: &mut Job<D>,
    cancel: &AtomicBool,
    deadline: Instant,
) -> Vec<Vec<usize>> {
    job.seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used = HashSet::new();
    let mut routes = vec![];
    for (_, seed) in &job.seeds {
        if stopped(cancel, deadline) || routes.len() == options.max_routes {
            break;
        }
        if seed.network != options.network || job.graph.exact_route_feasible(&seed.route).is_none()
        {
            continue;
        }
        let Some(customers) = seed.route.nodes[1..seed.route.nodes.len() - 1]
            .iter()
            .map(|&id| {
                usize::try_from(id)
                    .ok()
                    .filter(|&id| id > 0 && id < options.instance.customers.len())
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if customers.is_empty() || customers.iter().any(|id| used.contains(id)) {
            continue;
        }
        used.extend(customers.iter().copied());
        routes.push(customers);
    }
    routes
}

fn search_job<const D: usize>(
    options: &BackgroundIlsOptions,
    mut job: Job<D>,
    route_map: &HeuristicRouteMap,
    cancel: &AtomicBool,
    seed: i32,
) -> Vec<NetworkRoute> {
    let budget = Duration::from_secs_f64(options.search.time_limit);
    let deadline = Instant::now()
        .checked_add(budget)
        .unwrap_or(job.deadline)
        .min(job.deadline);
    let routes = neighborhood(options, &mut job, cancel, deadline);
    if routes.is_empty() || stopped(cancel, deadline) {
        return vec![];
    }
    let (instance, map, local_seed) =
        route_subproblem(&options.instance, routes.iter().map(Vec::as_slice));
    let mut search = options.search.clone();
    search.seed = seed;
    let Ok(result) =
        run_ails_ii_controlled(&instance, &local_seed, &search, Some(deadline), || {
            cancel.load(Ordering::Relaxed)
        })
    else {
        return vec![];
    };
    result
        .routes
        .iter()
        .filter(|r| !r.is_empty())
        .filter_map(|route| {
            let customers: Vec<_> = route.iter().map(|&id| map[id]).collect();
            if routes.contains(&customers) {
                return None;
            }
            Some(NetworkRoute {
                network: options.network,
                route: route_map.convert(&job.graph, &customers)?,
            })
        })
        .collect()
}

/// Recheck live branch/fixing domains, then rebuild all row coefficients and costs.
pub(super) fn collect<const D: usize>(
    state: &mut State<D>,
    routes: Vec<NetworkRoute>,
    duals: &[f64],
) -> Result<Vec<cg::Column>> {
    use crate::master::pricing::Networks;
    let mut columns = vec![];
    for route in routes {
        let Some(index) = state
            .oracles
            .iter()
            .position(|o| o.network.id == route.network)
        else {
            continue;
        };
        let oracle = &state.oracles[index];
        if state.universe.is_some()
            || state.gap_pool_cutoff.is_some()
            || oracle.solver.enumeration.is_some()
            || oracle.multiplicity_upper() == 0.
            || route.route.arcs.iter().any(|id| {
                oracle
                    .index
                    .get(id)
                    .is_none_or(|&p| oracle.proved_zero_arcs.get(p).copied().unwrap_or(false))
            })
            || !lifecycle::integer_route_feasible(oracle, &route.route)?
        {
            continue;
        }
        let tolerance = state.scheduler.tolerance;
        let collected = pricing::NetworkPricingAdapter { state, tolerance }.collect(
            index,
            vec![route.route],
            0.,
            duals,
        )?;
        columns.extend(collected.columns);
    }
    Ok(columns)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/background_ils_tests.rs"]
mod tests;
