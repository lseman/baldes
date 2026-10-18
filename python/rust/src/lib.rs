//! Python bindings for BALDES 2.0 — bucket-graph labeling RCSP solver.
//!
//! Uses the C ABI exposed by `baldes::api` to build a graph, configure resources
//! and arcs, then run pricing / enumeration.

use baldes::api::{
    or_rcsp_add_elementarity_set, or_rcsp_attach_elementarity_to_node,
    or_rcsp_build_default_resource_buckets, or_rcsp_delete, or_rcsp_enumerate,
    or_rcsp_get_enumerated, or_rcsp_get_enumerated_count, or_rcsp_get_option, or_rcsp_get_phase,
    or_rcsp_get_phase_count, or_rcsp_last_error, or_rcsp_new, or_rcsp_new_arc,
    or_rcsp_option_count, or_rcsp_option_name, or_rcsp_price, or_rcsp_set_arc_consumption,
    or_rcsp_set_bidirectional_mode, or_rcsp_set_graph_id, or_rcsp_set_initial_resource,
    or_rcsp_set_ng_neighborhood, or_rcsp_set_option, or_rcsp_set_pricing_controls,
    or_rcsp_set_resource_type, or_rcsp_set_sink, or_rcsp_set_source, or_rcsp_set_vertex_bounds,
    or_rcsp_solve, or_rcsp_update_reduced_costs, or_solution_delete, or_solution_get_arc_ids,
    or_solution_get_cost, or_solution_get_nb_nodes, or_solution_get_nodes,
    or_solution_get_res_consumption, or_solution_is_certified, ORPricingPhase, ORSolution, ORECSP,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::ffi::CString;

mod root;

// ── helpers ────────────────────────────────────────────────────────────────

fn last_error() -> PyErr {
    let msg = unsafe {
        let ptr = or_rcsp_last_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    PyRuntimeError::new_err(msg)
}

fn check_rc(ret: i32) -> PyResult<i32> {
    if ret == -1 {
        Err(last_error())
    } else {
        Ok(ret)
    }
}

/// Check the thread-local error state after a void-returning C call.
fn check_void() -> PyResult<()> {
    let err = or_rcsp_last_error();
    if err.is_null() {
        return Ok(());
    }
    // SAFETY: pointer is non-null; read as CStr.
    let cstr = unsafe { std::ffi::CStr::from_ptr(err) };
    let msg = cstr.to_string_lossy().into_owned();
    if msg.is_empty() {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(msg))
    }
}

// ── Opaque pointer wrappers (Send + Sync) ────────────────────────────────

/// A Send-safe wrapper around a raw `*mut ORECSP`.
struct SolverPtr(*mut ORECSP);
unsafe impl Send for SolverPtr {}
unsafe impl Sync for SolverPtr {}

/// A Send-safe wrapper around a raw `*mut ORSolution`.
struct RoutePtr(*mut ORSolution);
unsafe impl Send for RoutePtr {}
unsafe impl Sync for RoutePtr {}

// ── Route ──────────────────────────────────────────────────────────────────

/// A single elementary route (column) returned by pricing or enumeration.
#[pyclass(module = "pybaldes._core")]
struct Route {
    ptr: RoutePtr,
}

#[pymethods]
impl Route {
    #[getter]
    fn cost(&self) -> PyResult<f64> {
        let ret = unsafe { or_solution_get_cost(self.ptr.0) };
        if ret.is_nan() {
            Err(last_error())
        } else {
            Ok(ret)
        }
    }

    #[getter]
    fn nb_nodes(&self) -> PyResult<i32> {
        check_rc(unsafe { or_solution_get_nb_nodes(self.ptr.0) })
    }

    fn nodes(&self) -> PyResult<Vec<i32>> {
        let mut n: i32 = 0;
        let ptr = unsafe { or_solution_get_nodes(self.ptr.0, &mut n as *mut i32) };
        if ptr.is_null() {
            return Err(last_error());
        }
        // SAFETY: the returned pointer is valid for `n` elements.
        let slice = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        Ok(slice.to_vec())
    }

    fn arcs(&self) -> PyResult<Vec<i32>> {
        let mut n: i32 = 0;
        let ptr = unsafe { or_solution_get_arc_ids(self.ptr.0, &mut n as *mut i32) };
        if ptr.is_null() {
            return Err(last_error());
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        Ok(slice.to_vec())
    }

    fn resource_consumption(&self, resource_id: i32) -> PyResult<f64> {
        let ret = unsafe { or_solution_get_res_consumption(self.ptr.0, resource_id) };
        // The C API returns f64::NAN on error; detect NaN via self-subtraction.
        if (ret - ret).is_nan() {
            Err(last_error())
        } else {
            Ok(ret)
        }
    }

    #[getter]
    fn is_certified(&self) -> PyResult<bool> {
        check_rc(unsafe { or_solution_is_certified(self.ptr.0) }).map(|v| v != 0)
    }

    fn __repr__(&self) -> String {
        let cost = self.cost().unwrap_or(f64::NAN);
        let nb = self.nb_nodes().unwrap_or(0);
        format!("Route(cost={:.4}, nodes={nb})", cost)
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        if !self.ptr.0.is_null() {
            unsafe { or_solution_delete(self.ptr.0) };
        }
    }
}

// ── Phase ──────────────────────────────────────────────────────────────────

/// A pricing phase configuration.
#[pyclass(module = "pybaldes._core")]
#[derive(Clone)]
struct Phase {
    #[pyo3(get)]
    check_ng: bool,
    #[pyo3(get)]
    allow_jump_arcs: bool,
    #[pyo3(get)]
    labels_per_bucket: i64,
}

#[pymethods]
impl Phase {
    fn __repr__(&self) -> String {
        format!(
            "Phase(check_ng={}, jump_arcs={}, labels={})",
            self.check_ng, self.allow_jump_arcs, self.labels_per_bucket
        )
    }
}

// ── Solver ─────────────────────────────────────────────────────────────────

/// Stateful RCSP pricing solver.
///
/// Build a graph by adding vertices (implicitly via arc/source/sink),
/// resources, and arcs with costs and consumptions.  Then call
/// ``solve()`` for a single shortest route or ``price()`` / ``enumerate()``
/// for multiple columns.
#[pyclass(module = "pybaldes._core")]
struct Solver {
    ptr: SolverPtr,
    resource_counter: i32,
    n_vertices: usize,
}

#[pymethods]
impl Solver {
    // ── construction ───────────────────────────────────────────────────

    /// Create a new solver with ``n_vertices`` and ``n_resources``.
    #[new]
    fn new(n_vertices: i32, n_resources: i32) -> PyResult<Self> {
        let ptr = or_rcsp_new(n_vertices, n_resources);
        if ptr.is_null() {
            Err(last_error())
        } else {
            Ok(Solver {
                ptr: SolverPtr(ptr),
                resource_counter: 0,
                n_vertices: n_vertices as usize,
            })
        }
    }

    // ── graph topology ─────────────────────────────────────────────────

    /// Set the source vertex index.
    fn set_source(&mut self, v: i32) -> PyResult<()> {
        unsafe { or_rcsp_set_source(self.ptr.0, v) };
        check_void()
    }

    /// Set the sink vertex index.
    fn set_sink(&mut self, v: i32) -> PyResult<()> {
        unsafe { or_rcsp_set_sink(self.ptr.0, v) };
        check_void()
    }

    /// Set a custom graph ID (returned by some debug hooks).
    fn set_graph_id(&mut self, id: i32) -> PyResult<()> {
        unsafe { or_rcsp_set_graph_id(self.ptr.0, id) };
        check_void()
    }

    /// Add a directed arc from ``tail`` to ``head`` with the given cost.
    /// Returns the new arc's integer ID.
    fn new_arc(&mut self, tail: i32, head: i32, cost: f64) -> PyResult<i32> {
        check_rc(unsafe { or_rcsp_new_arc(self.ptr.0, tail, head, cost) })
    }

    // ── resources ──────────────────────────────────────────────────────

    /// Set the type of a pre-created resource and return its ID.
    ///
    /// Resources are pre-created when the solver is constructed with
    /// ``n_resources > 0``. This method sets the type (``0`` = disposable/
    /// cumulative, ``1`` = equality/reset) for the next resource index
    /// and returns that index.
    fn new_resource(&mut self, _name: &str, kind: i32) -> PyResult<i32> {
        let rid = self.resource_counter;
        check_rc(unsafe { or_rcsp_set_resource_type(self.ptr.0, rid, kind) })?;
        self.resource_counter += 1;
        Ok(rid)
    }

    /// Set the type of an existing resource (0 = disposable, 1 = equality).
    fn set_resource_type(&mut self, resource_id: i32, kind: i32) -> PyResult<()> {
        check_rc(unsafe { or_rcsp_set_resource_type(self.ptr.0, resource_id, kind) }).map(|_| ())
    }

    /// Set the fixed consumption of an arc for a given resource.
    fn set_arc_consumption(&mut self, arc_id: i32, resource_id: i32, value: f64) -> PyResult<()> {
        unsafe { or_rcsp_set_arc_consumption(self.ptr.0, arc_id, resource_id, value, value) };
        check_void()
    }

    /// Set the vertex time-window / capacity bounds for a resource.
    fn set_vertex_bounds(
        &mut self,
        vertex_id: i32,
        resource_id: i32,
        lower: f64,
        upper: f64,
    ) -> PyResult<()> {
        check_rc(unsafe {
            or_rcsp_set_vertex_bounds(self.ptr.0, vertex_id, resource_id, lower, upper)
        })
        .map(|_| ())
    }

    /// Set the initial resource value at the source.
    fn set_initial_resource(&mut self, resource_id: i32, value: f64) -> PyResult<()> {
        check_rc(unsafe { or_rcsp_set_initial_resource(self.ptr.0, resource_id, value) })
            .map(|_| ())
    }

    // ── NG neighbourhoods ──────────────────────────────────────────────

    /// Set the NG-neighbourhood for a vertex.  The list must include the
    /// vertex itself and have 1–64 entries.
    fn set_ng_neighborhood(&mut self, vertex_id: i32, nodes: Vec<i32>) -> PyResult<()> {
        check_rc(unsafe {
            or_rcsp_set_ng_neighborhood(self.ptr.0, vertex_id, nodes.as_ptr(), nodes.len() as i32)
        })
        .map(|_| ())
    }

    // ── elementarity sets ──────────────────────────────────────────────

    /// Add an elementarity set from a list of vertex IDs.
    fn add_elementarity_set(&mut self, nodes: Vec<i32>) -> PyResult<i32> {
        check_rc(unsafe {
            or_rcsp_add_elementarity_set(self.ptr.0, nodes.as_ptr(), nodes.len() as i32)
        })
    }

    /// Attach an elementarity set to a vertex.
    fn attach_elementarity_to_node(&mut self, vertex_id: i32, set_id: i32) -> PyResult<()> {
        check_rc(unsafe { or_rcsp_attach_elementarity_to_node(self.ptr.0, vertex_id, set_id) })
            .map(|_| ())
    }

    // ── pricing controls ───────────────────────────────────────────────

    /// Set pricing cutoff and resource limits.
    fn set_pricing_controls(
        &mut self,
        cutoff: f64,
        max_labels: usize,
        max_columns: usize,
    ) -> PyResult<()> {
        check_rc(unsafe {
            or_rcsp_set_pricing_controls(self.ptr.0, cutoff, max_labels, max_columns)
        })
        .map(|_| ())
    }

    /// Set a solver option. Unsupported options accept only their recovered default;
    /// see ``option_support(name)`` for availability and scope.
    fn set_option(&mut self, name: &str, value: f64) -> PyResult<()> {
        let cname = CString::new(name)
            .map_err(|_| PyValueError::new_err("option name must not contain null bytes"))?;
        check_rc(unsafe { or_rcsp_set_option(self.ptr.0, cname.as_ptr(), value) }).map(|_| ())
    }

    /// Get the current value of an option.
    fn get_option(&self, name: &str) -> PyResult<f64> {
        let mut out: f64 = 0.0;
        let cname = CString::new(name)
            .map_err(|_| PyValueError::new_err("option name must not contain null bytes"))?;
        check_rc(unsafe { or_rcsp_get_option(self.ptr.0, cname.as_ptr(), &mut out) })?;
        Ok(out)
    }

    /// Return (implemented, consumer or unsupported reason) for a recovered option.
    #[staticmethod]
    fn option_support(name: &str) -> PyResult<(bool, String)> {
        use baldes::core::options::{OptionSupport, Options};
        let support = Options::support(name).map_err(|e| PyValueError::new_err(e.message))?;
        Ok(match support {
            OptionSupport::Implemented(consumer) => (true, consumer.to_string()),
            OptionSupport::Unsupported(reason) => (false, reason.to_string()),
        })
    }

    /// Return the list of recognised option names.
    #[staticmethod]
    fn option_names() -> PyResult<Vec<String>> {
        let n = or_rcsp_option_count();
        let mut names = Vec::with_capacity(n as usize);
        for i in 0..n {
            let ptr = or_rcsp_option_name(i);
            if !ptr.is_null() {
                names.push(
                    unsafe { std::ffi::CStr::from_ptr(ptr) }
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        Ok(names)
    }

    /// Build default resource buckets (enables bucket-graph labeling).
    fn build_default_resource_buckets(&mut self) -> PyResult<()> {
        check_rc(unsafe { or_rcsp_build_default_resource_buckets(self.ptr.0) }).map(|_| ())
    }

    // ── reduced costs ──────────────────────────────────────────────────

    /// Update arc costs and vertex dual prices for column generation.
    fn update_reduced_costs(&mut self, costs: Vec<f64>, duals: Vec<f64>) -> PyResult<()> {
        if duals.len() != self.n_vertices {
            return Err(PyValueError::new_err(
                "duals must contain one value per vertex",
            ));
        }
        check_rc(unsafe {
            or_rcsp_update_reduced_costs(
                self.ptr.0,
                costs.as_ptr(),
                costs.len() as i32,
                duals.as_ptr(),
                std::ptr::null(),
                0,
            )
        })
        .map(|_| ())
    }

    // ── bidirectional search ───────────────────────────────────────────

    /// Set the bidirectional search mode: 0=off, 1=normal, 2=half-way.
    fn set_bidirectional_mode(&mut self, mode: i32) -> PyResult<()> {
        check_rc(unsafe { or_rcsp_set_bidirectional_mode(self.ptr.0, mode) }).map(|_| ())
    }

    // ── solve / price / enumerate ──────────────────────────────────────

    /// Run exact pricing for a single best route.
    ///
    /// Returns ``(cost, route)`` on success or ``None`` when no negative-cost
    /// path exists (pricing terminated).
    fn solve(&mut self) -> PyResult<Option<(f64, Route)>> {
        let mut cost: f64 = f64::INFINITY;
        let mut out: *mut ORSolution = std::ptr::null_mut();
        let ret = check_rc(unsafe { or_rcsp_solve(self.ptr.0, &mut cost, &mut out) })?;
        if ret == 0 {
            Ok(None)
        } else {
            let route = Route { ptr: RoutePtr(out) };
            Ok(Some((cost, route)))
        }
    }

    /// Run pricing to collect up to ``maximum`` columns.
    ///
    /// Returns ``(minimum_reduced_cost, [routes])``.  The route list may be
    /// empty when no negative-reduced-cost columns were found.
    fn price(&mut self, maximum: i32) -> PyResult<(f64, Vec<Route>)> {
        if maximum <= 0 {
            return Err(PyValueError::new_err("maximum must be positive"));
        }
        let mut routes = vec![std::ptr::null_mut(); maximum as usize];
        let mut minimum: f64 = f64::INFINITY;
        let mut exact: i32 = 0;
        let count = check_rc(unsafe {
            or_rcsp_price(
                self.ptr.0,
                routes.as_mut_ptr(),
                maximum,
                &mut minimum,
                &mut exact,
            )
        })?;
        let collected: Vec<Route> = routes
            .into_iter()
            .take(count as usize)
            .filter(|p| !p.is_null())
            .map(|p| Route { ptr: RoutePtr(p) })
            .collect();
        Ok((minimum, collected))
    }

    /// Run enumeration to collect up to ``maximum`` routes.
    fn enumerate(&mut self, maximum: i32) -> PyResult<(f64, i32, Vec<Route>)> {
        if maximum <= 0 {
            return Err(PyValueError::new_err("maximum must be positive"));
        }
        let mut routes = vec![std::ptr::null_mut(); maximum as usize];
        let minimum: f64 = 1e12;
        let exact: i32 = 0;
        let count =
            check_rc(unsafe { or_rcsp_enumerate(self.ptr.0, routes.as_mut_ptr(), maximum) })?;
        let collected: Vec<Route> = routes
            .into_iter()
            .take(count as usize)
            .filter(|p| !p.is_null())
            .map(|p| Route { ptr: RoutePtr(p) })
            .collect();
        Ok((minimum, exact, collected))
    }

    // ── phases ─────────────────────────────────────────────────────────

    /// Return the number of pricing phases in the current schedule.
    #[getter]
    fn phase_count(&self) -> PyResult<i32> {
        check_rc(unsafe { or_rcsp_get_phase_count(self.ptr.0) })
    }

    /// Get a phase by index.
    fn get_phase(&self, index: i32) -> PyResult<Phase> {
        let mut out = ORPricingPhase {
            check_ng_and_cuts: 0,
            allow_jump_arcs: 0,
            labels_per_bucket: 0,
        };
        check_rc(unsafe { or_rcsp_get_phase(self.ptr.0, index, &mut out) })?;
        Ok(Phase {
            check_ng: out.check_ng_and_cuts != 0,
            allow_jump_arcs: out.allow_jump_arcs != 0,
            labels_per_bucket: out.labels_per_bucket,
        })
    }

    // ── enumeration state ──────────────────────────────────────────────

    /// Number of routes currently stored in the enumeration universe.
    #[getter]
    fn enumerated_count(&self) -> PyResult<i32> {
        check_rc(unsafe { or_rcsp_get_enumerated_count(self.ptr.0) })
    }

    /// Retrieve an enumerated route by index.
    fn get_enumerated(&self, index: i32) -> PyResult<Route> {
        let ptr = unsafe { or_rcsp_get_enumerated(self.ptr.0, index) };
        if ptr.is_null() {
            Err(last_error())
        } else {
            Ok(Route { ptr: RoutePtr(ptr) })
        }
    }

    // ── lifecycle ──────────────────────────────────────────────────────

    fn __repr__(&self) -> String {
        if self.ptr.0.is_null() {
            "Solver(<deleted>)".to_string()
        } else {
            "Solver(<alive>)".to_string()
        }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        if !self.ptr.0.is_null() {
            unsafe { or_rcsp_delete(self.ptr.0) };
        }
    }
}

// ── module ─────────────────────────────────────────────────────────────────

/// BALDES 2.0 — bucket-graph labeling RCSP solver for vehicle routing.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<Route>()?;
    m.add_class::<Phase>()?;
    m.add("has_clp", cfg!(feature = "clp"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    root::register(m)?;
    Ok(())
}
