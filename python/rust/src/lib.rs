//! Python bindings backed directly by the Rust solver.
use baldes::pricing::Solver as RustSolver;
use baldes::ResourceKind;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "clp")]
mod mp;
mod root;
mod coupled;

fn run<T>(f: impl FnOnce() -> baldes::Result<T>) -> PyResult<T> {
    baldes::core::error::catch(f)
        .map_err(PyRuntimeError::new_err)?
        .map_err(|e| PyRuntimeError::new_err(e.message))
}
fn validate_option_name(name: &str) -> PyResult<()> {
    if name.contains('\0') {
        return Err(PyValueError::new_err(
            "option name must not contain null bytes",
        ));
    }
    Ok(())
}
fn index(value: i32) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyRuntimeError::new_err("negative count or ID"))
}
fn positive(value: i32) -> PyResult<usize> {
    if value <= 0 {
        return Err(PyValueError::new_err("maximum must be positive"));
    }
    Ok(value as usize)
}

/// An owned route that remains valid after its solver is dropped.
#[pyclass(module = "pybaldes._core")]
struct Route {
    route: baldes::Route,
    certified: bool,
}
#[pymethods]
impl Route {
    #[getter]
    fn cost(&self) -> f64 {
        self.route.cost
    }
    #[getter]
    fn original_cost(&self) -> f64 {
        self.route.original_cost
    }
    #[getter]
    fn nb_nodes(&self) -> usize {
        self.route.nodes.len()
    }
    fn nodes(&self) -> Vec<i32> {
        self.route.nodes.clone()
    }
    fn arcs(&self) -> Vec<i32> {
        self.route.arcs.clone()
    }
    fn resource_consumption(&self, resource_id: i32) -> PyResult<f64> {
        self.route
            .resources
            .get(index(resource_id)?)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err("solution resource out of range"))
    }
    #[getter]
    fn is_certified(&self) -> bool {
        self.certified
    }
    fn __repr__(&self) -> String {
        format!("Route(cost={:.4}, nodes={})", self.cost(), self.nb_nodes())
    }
}
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
/// Stateful RCSP solver owning its Rust graph and pricing state.
#[pyclass(module = "pybaldes._core")]
struct Solver {
    solver: std::sync::Mutex<RustSolver>,
    resource_counter: i32,
}
impl Solver {
    fn resource(&self, id: i32) -> PyResult<usize> {
        let id = index(id)?;
        if id >= self.solver.lock().unwrap().graph.resources.len() {
            return Err(PyRuntimeError::new_err("resource out of range"));
        }
        Ok(id)
    }
    fn vertex(&self, id: i32) -> PyResult<usize> {
        let id = index(id)?;
        if id >= self.solver.lock().unwrap().graph.vertices.len() {
            return Err(PyRuntimeError::new_err("vertex out of range"));
        }
        Ok(id)
    }
}
#[pymethods]
impl Solver {
    #[new]
    /// Create a new solver with ``n_vertices`` and ``n_resources``.
    fn new(n_vertices: i32, n_resources: i32) -> PyResult<Self> {
        let n = index(n_vertices)?;
        let r = index(n_resources)?;
        Ok(Self {
            solver: std::sync::Mutex::new(run(|| RustSolver::new(n, r))?),
            resource_counter: 0,
        })
    }
    /// Set the source vertex index.
    fn set_source(&mut self, v: i32) -> PyResult<()> {
        let v = self.vertex(v)?;
        self.solver.get_mut().unwrap().graph.source = v;
        Ok(())
    }
    /// Set the sink vertex index.
    fn set_sink(&mut self, v: i32) -> PyResult<()> {
        let v = self.vertex(v)?;
        self.solver.get_mut().unwrap().graph.sink = v;
        Ok(())
    }
    /// Set a custom graph ID (returned by some debug hooks).
    fn set_graph_id(&mut self, id: i32) {
        self.solver.get_mut().unwrap().graph.graph_id = id;
    }
    /// Add a directed arc from ``tail`` to ``head`` with the given cost.
    /// Returns the new arc's integer ID.
    fn new_arc(&mut self, tail: i32, head: i32, cost: f64) -> PyResult<i32> {
        let tail = index(tail)?;
        let head = index(head)?;
        run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .add_arc(tail, head, cost)
        })
    }
    /// Set the type of a pre-created resource and return its ID.
    ///
    /// Resources are pre-created when the solver is constructed with
    /// ``n_resources > 0``. This method sets the type (``0`` = disposable/
    /// cumulative, ``1`` = equality/reset) for the next resource index
    /// and returns that index.
    fn new_resource(&mut self, _name: &str, kind: i32) -> PyResult<i32> {
        let rid = self.resource_counter;
        self.set_resource_type(rid, kind)?;
        self.resource_counter += 1;
        Ok(rid)
    }
    /// Set the type of an existing resource (0 = disposable, 1 = equality).
    fn set_resource_type(&mut self, resource_id: i32, kind: i32) -> PyResult<()> {
        let r = self.resource(resource_id)?;
        self.solver.get_mut().unwrap().graph.resources[r] = match kind {
            0 => ResourceKind::Disposable,
            1 => ResourceKind::Equality,
            _ => return Err(PyRuntimeError::new_err("invalid resource type")),
        };
        Ok(())
    }
    /// Set the fixed consumption of an arc for a given resource.
    fn set_arc_consumption(&mut self, arc_id: i32, resource_id: i32, value: f64) -> PyResult<()> {
        let r = self.resource(resource_id)?;
        if !value.is_finite() {
            return Err(PyRuntimeError::new_err("finite consumption required"));
        }
        run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .arc_mut(arc_id)?
                .consumption[r] = value;
            Ok(())
        })
    }
    /// Set the vertex time-window / capacity bounds for a resource.
    fn set_vertex_bounds(
        &mut self,
        vertex_id: i32,
        resource_id: i32,
        lower: f64,
        upper: f64,
    ) -> PyResult<()> {
        let v = index(vertex_id)?;
        let r = self.resource(resource_id)?;
        run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .set_bounds(v, r, lower, upper)
        })
    }
    /// Set disjoint closed service-start windows on a disposable resource.
    /// Requires the ``multi-time-windows`` build feature.
    #[cfg(feature = "multi-time-windows")]
    fn set_vertex_time_windows(
        &mut self,
        vertex_id: i32,
        resource_id: i32,
        windows: Vec<(f64, f64)>,
    ) -> PyResult<()> {
        let v = self.vertex(vertex_id)?;
        let r = self.resource(resource_id)?;
        run(|| {
            let windows = windows
                .into_iter()
                .map(|(a, b)| baldes::graph::Window::new(a, b))
                .collect::<baldes::Result<Vec<_>>>()?;
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .set_time_windows(v, r, &windows)
        })
    }
    /// Set the initial resource value at the source.
    fn set_initial_resource(&mut self, resource_id: i32, value: f64) -> PyResult<()> {
        let r = self.resource(resource_id)?;
        if !value.is_finite() {
            return Err(PyRuntimeError::new_err("finite initial resource required"));
        }
        self.solver.get_mut().unwrap().graph.initial[r] = Some(value);
        Ok(())
    }
    /// Set the NG-neighbourhood for a vertex.  The list must include the
    /// vertex itself and have 1–64 entries.
    fn set_ng_neighborhood(&mut self, vertex_id: i32, nodes: Vec<i32>) -> PyResult<()> {
        let v = self.vertex(vertex_id)?;
        if nodes.is_empty() || nodes.len() > 64 || !nodes.contains(&vertex_id) {
            return Err(PyRuntimeError::new_err(
                "NG neighborhood must contain its vertex and have 1..64 entries",
            ));
        }
        let ids = nodes.into_iter().map(index).collect::<PyResult<Vec<_>>>()?;
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != ids.len() {
            return Err(PyRuntimeError::new_err("duplicate NG vertex"));
        }
        run(|| self.solver.get_mut().unwrap().graph.set_ng(v, &ids))
    }
    /// Add an elementarity set from a list of vertex IDs.
    fn add_elementarity_set(&mut self, nodes: Vec<i32>) -> PyResult<i32> {
        let ids = nodes.into_iter().map(index).collect::<PyResult<Vec<_>>>()?;
        run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .add_elementarity_set(&ids)
        })
        .map(|id| id as i32)
    }
    /// Attach an elementarity set to a vertex.
    fn attach_elementarity_to_node(&mut self, vertex_id: i32, set_id: i32) -> PyResult<()> {
        let v = self.vertex(vertex_id)?;
        let set = index(set_id)?;
        if set >= self.solver.get_mut().unwrap().graph.element_count {
            return Err(PyRuntimeError::new_err(
                "elementarity set/node out of range",
            ));
        }
        let elements = &mut self.solver.get_mut().unwrap().graph.vertices[v].elements;
        if !elements.contains(&set) {
            elements.push(set);
        }
        Ok(())
    }
    /// Set pricing cutoff and resource limits.
    fn set_pricing_controls(
        &mut self,
        cutoff: f64,
        max_labels: usize,
        max_columns: usize,
    ) -> PyResult<()> {
        if cutoff.is_nan() || max_labels == 0 || max_columns == 0 {
            return Err(PyRuntimeError::new_err("pricing controls"));
        }
        self.solver.get_mut().unwrap().cutoff = cutoff;
        self.solver.get_mut().unwrap().controls.max_labels = max_labels;
        self.solver.get_mut().unwrap().controls.max_columns = max_columns;
        Ok(())
    }
    /// Set a BALDES solver option.
    /// See ``option_support(name)`` for availability and scope.
    fn set_option(&mut self, name: &str, value: f64) -> PyResult<()> {
        validate_option_name(name)?;
        run(|| self.solver.get_mut().unwrap().set_option(name, value))
    }
    /// Get the current value of an option.
    fn get_option(&self, name: &str) -> PyResult<f64> {
        validate_option_name(name)?;
        run(|| self.solver.lock().unwrap().options.get(name))
    }
    #[staticmethod]
    /// Return availability and scope for an explicit BALDES option query.
    fn option_support(name: &str) -> PyResult<(bool, String)> {
        use baldes::core::options::{OptionSupport, Options};
        let support = Options::support(name).map_err(|e| PyValueError::new_err(e.message))?;
        Ok(match support {
            OptionSupport::Implemented(consumer) => (true, consumer.to_string()),
            OptionSupport::Unsupported(reason) => (false, reason.to_string()),
        })
    }
    #[staticmethod]
    /// Return supported BALDES option names for the current build.
    fn option_names() -> Vec<String> {
        baldes::core::options::OPTION_TABLE
            .iter()
            .map(|info| info.name.to_string())
            .collect()
    }
    /// Build default resource buckets (enables bucket-graph labeling).
    fn build_default_resource_buckets(&mut self) -> PyResult<()> {
        use baldes::labeling::engine::{EngineControls, LabelingEngine};
        use baldes::pricing::phases::Phase;
        run(|| {
            let solver = self.solver.get_mut().unwrap();
            solver.bucket_steps =
                baldes::graph::ResourceValues::filled(solver.graph.resources.len(), 0.);
            let controls = EngineControls {
                steps: solver.bucket_steps,
                bidirectional: solver.options.get("BALDESuseBidirectionalSearch")? != 0.,
                target_buckets: solver.options.get("BALDESnumberOfBucketsPerVertex")? as i32,
                ..Default::default()
            };
            solver.engine = Some(LabelingEngine::prepare(
                &solver.graph,
                controls,
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )?);
            Ok(())
        })
    }
    /// Update arc costs and vertex dual prices for column generation.
    fn update_reduced_costs(&mut self, costs: Vec<f64>, duals: Vec<f64>) -> PyResult<()> {
        if duals.len() != self.solver.get_mut().unwrap().graph.vertices.len() {
            return Err(PyValueError::new_err(
                "duals must contain one value per vertex",
            ));
        }
        run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .graph
                .update_reduced_costs(&costs, &duals, false)
        })
    }
    /// Set the bidirectional search mode: 0=off, 1=normal, 2=half-way.
    fn set_bidirectional_mode(&mut self, mode: i32) -> PyResult<()> {
        self.set_option("BALDESuseBidirectionalSearch", mode as f64)
    }
    /// Run exact pricing for a single best route.
    ///
    /// Returns ``(cost, route)`` on success or ``None`` when no negative-cost
    /// path exists (pricing terminated).
    fn solve(&mut self) -> PyResult<Option<(f64, Route)>> {
        let result = run(|| self.solver.get_mut().unwrap().price(1, false))?;
        Ok(result.columns.into_iter().next().map(|route| {
            (
                route.cost,
                Route {
                    route,
                    certified: result.exact,
                },
            )
        }))
    }
    /// Run pricing to collect up to ``maximum`` columns.
    ///
    /// Returns ``(minimum_reduced_cost, [routes])``.  The route list may be
    /// empty when no negative-reduced-cost columns were found.
    fn price(&mut self, maximum: i32) -> PyResult<(f64, Vec<Route>)> {
        let maximum = positive(maximum)?;
        let result = run(|| self.solver.get_mut().unwrap().price(maximum, true))?;
        Ok((
            result.minimum_reduced_cost,
            result
                .columns
                .into_iter()
                .map(|route| Route {
                    route,
                    certified: result.exact,
                })
                .collect(),
        ))
    }
    /// Run enumeration to collect up to ``maximum`` routes.
    fn enumerate(&mut self, maximum: i32) -> PyResult<(f64, i32, Vec<Route>)> {
        let maximum = positive(maximum)?;
        let result = run(|| {
            self.solver
                .get_mut()
                .unwrap()
                .price_phase(0, maximum, false)
        })?;
        Ok((
            1e12,
            0,
            result
                .columns
                .into_iter()
                .map(|route| Route {
                    route,
                    certified: result.exact,
                })
                .collect(),
        ))
    }
    #[getter]
    /// Return the number of pricing phases in the current schedule.
    fn phase_count(&self) -> usize {
        self.solver.lock().unwrap().phases.len()
    }
    /// Get a phase by index.
    fn get_phase(&self, index: i32) -> PyResult<Phase> {
        let index =
            usize::try_from(index).map_err(|_| PyRuntimeError::new_err("negative count or ID"))?;
        let solver = self.solver.lock().unwrap();
        let phase = solver
            .phases
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err("phase out of range"))?;
        Ok(Phase {
            check_ng: phase.check_ng,
            allow_jump_arcs: phase.jump_arcs,
            labels_per_bucket: phase.labels_per_bucket,
        })
    }
    #[getter]
    /// Number of routes currently stored in the enumeration universe.
    fn enumerated_count(&self) -> usize {
        self.solver
            .lock()
            .unwrap()
            .enumeration
            .as_ref()
            .map_or(0, |e| e.paths.len)
    }
    /// Retrieve an enumerated route by index.
    fn get_enumerated(&self, index: i32) -> PyResult<Route> {
        let index =
            usize::try_from(index).map_err(|_| PyRuntimeError::new_err("negative count or ID"))?;
        let solver = self.solver.lock().unwrap();
        let universe = solver
            .enumeration
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no enumerated route universe"))?;
        let route = run(|| universe.paths.routes(&solver.graph))?
            .into_iter()
            .nth(index)
            .ok_or_else(|| PyRuntimeError::new_err("enumerated route index out of range"))?;
        Ok(Route {
            route,
            certified: true,
        })
    }
    fn __repr__(&self) -> &'static str {
        "Solver(<alive>)"
    }
}

// ── module ─────────────────────────────────────────────────────────────────

/// BALDES 2.0 — bucket-graph labeling RCSP solver for vehicle routing.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solver>()?;
    m.add_class::<Route>()?;
    m.add_class::<Phase>()?;
    coupled::register(m)?;
    m.add("resource_dimensions", baldes::DEFAULT_RESOURCE_DIMENSIONS)?;
    m.add("has_clp", cfg!(feature = "clp"))?;
    m.add("has_highs", cfg!(feature = "highs"))?;
    m.add("has_gurobi", cfg!(feature = "gurobi"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    root::register(m)?;
    #[cfg(feature = "clp")]
    mp::register(m)?;
    Ok(())
}
