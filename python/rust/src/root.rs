//! Owned Python inputs and results for the Rust routing front end.
use baldes::inputs::Instance;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

type CustomerData = (f64, f64, f64, f64, f64, f64);

#[pyfunction]
fn _read_instance(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let instance = Instance::read(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let data = PyDict::new(py);
    data.set_item("name", instance.name)?;
    data.set_item("vehicles", instance.vehicles)?;
    data.set_item("capacity", instance.capacity)?;
    let customers: Vec<CustomerData> = instance
        .customers
        .iter()
        .map(|c| (c.x, c.y, c.demand, c.ready, c.due, c.service))
        .collect();
    data.set_item("customers", customers)?;
    let rounding = match instance.rounding {
        baldes::inputs::Rounding::None => "none",
        baldes::inputs::Rounding::Nearest => "nearest",
        baldes::inputs::Rounding::Tenth => "tenth",
        baldes::inputs::Rounding::Ceil => "ceil",
    };
    data.set_item("rounding", rounding)?;
    data.set_item("distances", instance.distances)?;
    Ok(data.unbind())
}

#[cfg(feature = "clp")]
#[pyfunction]
#[pyo3(signature = (name, vehicles, capacity, customers, distances, problem_type, ng, reference_ng, rounding, capacity_cuts, rank_one_cuts, stabilize, enumerate, upper_bound, time_limit, max_iterations, max_cut_rounds, pricing_options, two_path_cuts=false, master_solver="clp", hgs_time=None, hgs_columns="best", hgs_seed=0, verbose=false, log_interval=10))]
#[allow(clippy::too_many_arguments)]
fn _solve_root(
    py: Python<'_>,
    name: String,
    vehicles: usize,
    capacity: f64,
    customers: Vec<CustomerData>,
    distances: Option<Vec<Vec<f64>>>,
    problem_type: &str,
    ng: usize,
    reference_ng: bool,
    rounding: &str,
    capacity_cuts: bool,
    rank_one_cuts: bool,
    stabilize: bool,
    enumerate: bool,
    upper_bound: f64,
    time_limit: f64,
    max_iterations: usize,
    max_cut_rounds: Option<usize>,
    pricing_options: std::collections::BTreeMap<String, f64>,
    two_path_cuts: bool,
    master_solver: &str,
    hgs_time: Option<f64>,
    hgs_columns: &str,
    hgs_seed: i32,
    verbose: bool,
    log_interval: usize,
) -> PyResult<Py<PyDict>> {
    use baldes::master::root::{self, RootOptions, RootProblem};
    use baldes::inputs::{Customer, Rounding};
    if problem_type != "vrptw" && problem_type != "cvrp" {
        return Err(PyValueError::new_err(
            "problem_type must be 'vrptw' or 'cvrp'",
        ));
    }
    if two_path_cuts && problem_type == "cvrp" {
        return Err(PyValueError::new_err("Kohl 2-path cuts require vrptw time windows"));
    }
    let rounding = Rounding::parse(rounding)
        .map_err(|_| PyValueError::new_err("rounding must be 'none', 'nearest', 'tenth' or 'ceil'"))?;
    if let Some(matrix) = &distances {
        if matrix.len() != customers.len()
            || matrix.iter().any(|row| row.len() != customers.len() || row.iter().any(|&d| !(d >= 0. && d.is_finite())))
        {
            return Err(PyValueError::new_err(
                "distances must be a square matrix over the depot and customers, finite and nonnegative",
            ));
        }
    }
    if ng == 0
        || max_iterations == 0
        || time_limit <= 0.
        || !time_limit.is_finite()
        || upper_bound.is_nan()
        || upper_bound <= 0.
    {
        return Err(PyValueError::new_err("ng, max_iterations, time_limit and upper_bound must be positive; time_limit must be finite"));
    }
    if vehicles == 0 || capacity <= 0. || !capacity.is_finite() || customers.len() < 2 {
        return Err(PyValueError::new_err(
            "instance needs positive vehicles/capacity, a depot and at least one customer",
        ));
    }
    if problem_type == "vrptw" && customers.iter().any(|c| c.4.is_infinite()) {
        return Err(PyValueError::new_err(
            "the instance has no time windows (e.g. CVRPLIB input); use problem_type='cvrp'",
        ));
    }
    // CVRP has no time resource constraints, including when callers omit due times.
    let customers: Vec<_> = customers
        .into_iter()
        .map(|(x, y, demand, ready, due, service)| {
            if problem_type == "cvrp" {
                (x, y, demand, 0., 1., 0.)
            } else {
                (x, y, demand, ready, due, service)
            }
        })
        .collect();
    for &(x, y, demand, ready, due, service) in &customers {
        if [x, y, demand, ready, due, service]
            .iter()
            .any(|v| !v.is_finite())
            || demand < 0.
            || ready > due
            || service < 0.
        {
            return Err(PyValueError::new_err(
                "invalid customer data; VRPTW needs finite coordinates, demand, time windows and service times",
            ));
        }
    }
    if customers[0].2 != 0. {
        return Err(PyValueError::new_err("depot demand must be zero"));
    }
    let instance = Instance {
        name,
        vehicles,
        capacity,
        customers: customers
            .into_iter()
            .map(|(x, y, demand, ready, due, service)| Customer {
                x,
                y,
                demand,
                ready,
                due,
                service,
            })
            .collect(),
        rounding,
        distances,
    };
    let mut options = RootOptions::default();
    if log_interval == 0 { return Err(PyValueError::new_err("log_interval must be positive")); }
    options.cg.log_interval = verbose.then_some(log_interval);
    options.master_solver = match master_solver {
        "clp" => root::MasterSolver::Clp,
        "ipm" => root::MasterSolver::Ipm,
        "highs" => root::MasterSolver::Highs,
        _ => return Err(PyValueError::new_err("master_solver must be 'clp', 'highs', or 'ipm'")),
    };
    options.master_solver.ensure_available().map_err(|e|PyValueError::new_err(e.to_string()))?;
    let columns = baldes::inputs::HgsColumns::parse(hgs_columns)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let warm = hgs_time.map(|seconds| {
        let time_limit = std::time::Duration::try_from_secs_f64(seconds)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if time_limit.is_zero() { return Err(PyValueError::new_err("hgs_time must be positive")); }
        Ok(baldes::inputs::HgsWarmStart { time_limit, columns, seed: hgs_seed })
    }).transpose()?;
    options.cg.stabilize = stabilize;
    options.cg.max_iterations = max_iterations;
    options.cg.time_limit = std::time::Duration::try_from_secs_f64(time_limit)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    options.enumerate = enumerate;
    options.incumbent = upper_bound;
    options.pricing_options = pricing_options.into_iter().collect();
    if rank_one_cuts {
        options = options.with_rank_one_cuts();
    }
    if capacity_cuts || two_path_cuts {
        options = options.with_capacity_cuts(instance.capacity_cut_config(rounding, two_path_cuts));
    }
    if let Some(limit) = max_cut_rounds {
        options.cg.max_cut_rounds = limit;
    }
    let cvrp = problem_type == "cvrp";
    // All inputs and the result own their storage. Python can run other threads
    // while the Rust solver and its private Clp master execute.
    let started = std::time::Instant::now();
    // Node snapshots retain an Rc<ClpMaster> and must stay on the solver
    // thread. Python consumes only the owned result data, which is Send.
    struct PythonRootOutput {
        master: baldes::master::column_generation::Output,
        routes: Vec<baldes::Route>,
        values: Vec<f64>,
        duals: Vec<f64>,
        incumbent: f64,
        pricing_seconds: f64,
        solver_seconds: f64,
        cuts: usize,
        lp_before_cuts: f64,
        fixing_calls: usize,
        enumerated_routes: usize,
    }
    let output = py
        .allow_threads(move || -> baldes::Result<_> {
            let mut graph = if cvrp {
                instance.capacity_graph(ng, rounding)?
            } else {
                instance.graph(ng, rounding)?
            };
            let n = instance.customers.len() - 1;
            if reference_ng {
                for v in 0..=n + 1 {
                    let mut neighbors = if v == 0 || v == n + 1 {
                        vec![v]
                    } else {
                        let mut ids: Vec<_> = (1..=n).collect();
                        ids.sort_by(|&a, &b| {
                            instance
                                .distance(v, a, rounding)
                                .total_cmp(&instance.distance(v, b, rounding))
                                .then(a.cmp(&b))
                        });
                        ids.truncate(ng);
                        ids
                    };
                    neighbors.sort_unstable();
                    let elements: Vec<_> = neighbors.iter()
                        .flat_map(|&vertex| graph.vertices[vertex].elements.iter().copied()).collect();
                    graph.set_ng(v, &elements)?;
                }
            }
            options.cg.multiplicity = n as f64;
            let problem = RootProblem::set_partitioning(graph, n as f64)?;
            if let Some(warm) = warm {
                instance.seed_hgs(&problem, rounding, &mut options, &warm)?;
            }
            let output = root::solve(problem, options)?;
            Ok(PythonRootOutput {
                master: output.master, routes: output.routes, values: output.values,
                duals: output.duals, incumbent: output.incumbent,
                pricing_seconds: output.pricing_seconds, solver_seconds: output.solver_seconds,
                cuts: output.cuts, lp_before_cuts: output.lp_before_cuts,
                fixing_calls: output.fixing_calls, enumerated_routes: output.enumerated_routes,
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let data = PyDict::new(py);
    macro_rules! field {
        ($key:literal, $value:expr) => {
            data.set_item($key, $value)?;
        };
    }
    field!("status", format!("{:?}", output.master.status));
    field!("lp", output.master.lp);
    field!("bound", output.master.bound);
    field!("lp_before_cuts", output.lp_before_cuts);
    field!("incumbent", output.incumbent);
    field!("cuts", output.cuts);
    field!("cut_rounds", output.master.cut_rounds);
    field!("iterations", output.master.iterations);
    field!("pricing_calls", output.master.pricing_calls);
    // Generated columns inserted over the run.
    field!("columns", output.master.inserted_tags.len());
    field!("master_columns", output.master.columns);
    field!("fixing_calls", output.fixing_calls);
    field!("enumerated_routes", output.enumerated_routes);
    field!("pruned_by_bound", output.master.pruned_by_bound);
    field!("duals", output.duals);
    field!("seconds", started.elapsed().as_secs_f64());
    field!("pricing_seconds", output.pricing_seconds);
    field!("solver_seconds", output.solver_seconds);
    let mut routes = Vec::new();
    for (&tag, &value) in output.master.column_tags.iter().zip(&output.values) {
        if tag < 0 || value <= 1e-9 {
            continue;
        }
        let route = &output.routes[tag as usize];
        let item = PyDict::new(py);
        item.set_item("value", value)?;
        item.set_item("cost", route.original_cost)?;
        item.set_item("nodes", &route.nodes)?;
        item.set_item("arcs", &route.arcs)?;
        routes.push(item);
    }
    field!("routes", routes);
    let mut history = Vec::new();
    for h in output.master.history {
        let item = PyDict::new(py);
        item.set_item("iteration", h.iteration)?;
        item.set_item("lp", h.lp)?;
        item.set_item("minimum_reduced_cost", h.minimum)?;
        item.set_item("new_columns", h.new_columns)?;
        item.set_item("columns", h.columns)?;
        item.set_item("exact", h.exact)?;
        item.set_item("phase", h.phase)?;
        item.set_item("pricing_calls", h.pricing_calls)?;
        item.set_item("alpha", h.alpha)?;
        item.set_item("misprices", h.misprices)?;
        history.push(item);
    }
    field!("history", history);
    Ok(data.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_read_instance, m)?)?;
    #[cfg(feature = "clp")]
    m.add_function(wrap_pyfunction!(_solve_root, m)?)?;
    Ok(())
}
