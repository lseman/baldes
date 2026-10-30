//! Direct coupled-resource model installation for the low-level Python API.
use super::{run, Solver};
use baldes::graph::coupled::{
    descriptor::RawAffine, CoupledResource, EncodedFunction as FunctionDefinition,
    EvaluationPolicy, FrontOperationPolicy, PairedResourceMode, PiecewiseLinearFunction,
    PreparedMachineArc, PreparedMachineBackwardInputs, PreparedSlopeFunction,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Explicit function encodings retain their own evaluation order and policy.
#[derive(Clone)]
#[pyclass(frozen, module = "pybaldes._core")]
pub struct EncodedFunction {
    definition: FunctionDefinition,
}

fn evaluation_policy(policy: &str) -> PyResult<EvaluationPolicy> {
    match policy {
        "strict_domain" => Ok(EvaluationPolicy::StrictDomain),
        "endpoint_tolerance" => Ok(EvaluationPolicy::EndpointTolerance),
        _ => Err(PyValueError::new_err(
            "policy must be 'strict_domain' or 'endpoint_tolerance'",
        )),
    }
}

fn front_policy(policy: &str) -> PyResult<(FrontOperationPolicy, EvaluationPolicy)> {
    match policy {
        "strict_domain" => Ok((
            FrontOperationPolicy::StrictDomain,
            EvaluationPolicy::StrictDomain,
        )),
        "front_algebra" => Ok((
            FrontOperationPolicy::FrontAlgebra,
            EvaluationPolicy::EndpointTolerance,
        )),
        _ => Err(PyValueError::new_err(
            "front policy must be 'strict_domain' or 'front_algebra'",
        )),
    }
}

fn slope_function(
    definition: &FunctionDefinition,
    expected: EvaluationPolicy,
) -> PyResult<PreparedSlopeFunction> {
    match definition {
        FunctionDefinition::SlopeSegments { function, policy } if *policy == expected => Ok(function.clone()),
        FunctionDefinition::SlopeSegments { .. } => Err(PyValueError::new_err("slope function evaluation policy must match front policy: strict_domain requires strict_domain; front_algebra requires endpoint_tolerance")),
        FunctionDefinition::Knots { .. } => Err(PyValueError::new_err("slope resources require slope_segments functions; knots cannot be converted losslessly")),
    }
}

fn exact_fields(dictionary: &Bound<'_, PyDict>, expected: &[&str]) -> PyResult<()> {
    if dictionary.len() != expected.len()
        || expected
            .iter()
            .any(|key| !dictionary.contains(key).unwrap_or(false))
    {
        return Err(PyValueError::new_err(format!(
            "expected exactly these fields: {}",
            expected.join(", ")
        )));
    }
    Ok(())
}

fn required_field<'py>(dictionary: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    dictionary
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing field: {name}")))
}

/// JSON encodes pairs as lists; accept both lists and tuples with exactly two values.
fn wire_pairs(value: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let pairs = value
        .extract::<Vec<[f64; 2]>>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(pairs
        .into_iter()
        .map(|[first, second]| (first, second))
        .collect())
}

fn prepared_machine_function(
    dictionary: &Bound<'_, PyDict>,
    name: &str,
) -> PyResult<PreparedSlopeFunction> {
    let value = required_field(dictionary, name)?;
    let function = value.cast::<PyDict>()?;
    exact_fields(function, &["pairs"])?;
    PreparedSlopeFunction::from_machine_pairs(wire_pairs(&required_field(function, "pairs")?)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn prepared_machine_arc(dictionary: &Bound<'_, PyDict>) -> PyResult<PreparedMachineArc> {
    let mut fields = vec![
        "tail_limit",
        "head_limit",
        "waiting_lower",
        "waiting_upper",
        "waiting_rate",
        "forward_mapping",
        "inverse",
        "sum_vector",
        "dependent_lower",
        "dependent_upper",
    ];
    if dictionary.contains("backward")? {
        fields.push("backward");
    }
    exact_fields(dictionary, &fields)?;
    let backward = match dictionary.get_item("backward")? {
        Some(value) if !value.is_none() => {
            let data = value.cast::<PyDict>()?;
            exact_fields(
                data,
                &[
                    "tail_lower",
                    "tail_upper",
                    "forward_at_tail_lower",
                    "mapped_upper",
                    "dependent",
                ],
            )?;
            Some(PreparedMachineBackwardInputs {
                tail_lower: required_field(data, "tail_lower")?.extract()?,
                tail_upper: required_field(data, "tail_upper")?.extract()?,
                forward_at_tail_lower: required_field(data, "forward_at_tail_lower")?.extract()?,
                mapped_upper: required_field(data, "mapped_upper")?.extract()?,
                dependent: prepared_machine_function(data, "dependent")?,
            })
        }
        _ => None,
    };
    Ok(PreparedMachineArc {
        tail_limit: required_field(dictionary, "tail_limit")?.extract()?,
        head_limit: required_field(dictionary, "head_limit")?.extract()?,
        waiting_lower: required_field(dictionary, "waiting_lower")?.extract()?,
        waiting_upper: required_field(dictionary, "waiting_upper")?.extract()?,
        waiting_rate: required_field(dictionary, "waiting_rate")?.extract()?,
        forward_mapping: prepared_machine_function(dictionary, "forward_mapping")?,
        inverse: prepared_machine_function(dictionary, "inverse")?,
        sum_vector: prepared_machine_function(dictionary, "sum_vector")?,
        dependent_lower: prepared_machine_function(dictionary, "dependent_lower")?,
        dependent_upper: prepared_machine_function(dictionary, "dependent_upper")?,
        backward,
    })
}

#[derive(Clone)]
enum FrontInput {
    Affine(RawAffine),
    Vector(PreparedSlopeFunction),
}
impl FrontInput {
    fn input(&self) -> baldes::graph::coupled::descriptor::MachineRootInput<'_> {
        use baldes::graph::coupled::descriptor::MachineRootInput;
        match self {
            Self::Affine(value) => MachineRootInput::Affine(*value),
            Self::Vector(value) => MachineRootInput::Vector(value),
        }
    }
}

fn front_input(dictionary: &Bound<'_, PyDict>) -> PyResult<FrontInput> {
    let encoding: String = required_field(dictionary, "encoding")?.extract()?;
    match encoding.as_str() {
        "affine" => {
            exact_fields(dictionary, &["encoding", "intercept", "slope"])?;
            let intercept = required_field(dictionary, "intercept")?.extract()?;
            let slope = required_field(dictionary, "slope")?.extract()?;
            Ok(FrontInput::Affine(
                RawAffine::new(intercept, slope)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ))
        }
        "vector" => {
            exact_fields(dictionary, &["encoding", "function"])?;
            Ok(FrontInput::Vector(prepared_machine_function(
                dictionary, "function",
            )?))
        }
        _ => Err(PyValueError::new_err(
            "front input encoding must be affine or vector",
        )),
    }
}

#[derive(Clone)]
struct FrontArc {
    tail: baldes::graph::Window,
    waiting: baldes::graph::Window,
    rate: f64,
    base: PreparedSlopeFunction,
    dependent: FrontInput,
    lower: FrontInput,
    upper: FrontInput,
}
#[derive(Clone)]
struct FrontProgram {
    source: baldes::graph::Window,
    terminal: baldes::graph::Window,
    root: FrontInput,
    arcs: Vec<FrontArc>,
}

fn front_window(value: Bound<'_, PyAny>) -> PyResult<baldes::graph::Window> {
    let [lower, upper] = value.extract::<[f64; 2]>()?;
    baldes::graph::Window::new(lower, upper).map_err(|error| PyValueError::new_err(error.message))
}

fn front_program(dictionary: &Bound<'_, PyDict>) -> PyResult<FrontProgram> {
    exact_fields(
        dictionary,
        &["source_bounds", "terminal_bounds", "root", "arcs"],
    )?;
    let py = dictionary.py();
    let arcs = required_field(dictionary, "arcs")?.extract::<Vec<Py<PyDict>>>()?;
    let mut parsed = Vec::with_capacity(arcs.len());
    for arc in arcs {
        let arc = arc.bind(py);
        exact_fields(
            arc,
            &[
                "tail_bounds",
                "waiting_bounds",
                "waiting_rate",
                "base",
                "dependent",
                "dependent_lower",
                "dependent_upper",
            ],
        )?;
        let rate: f64 = required_field(arc, "waiting_rate")?.extract()?;
        if !rate.is_finite()
            || rate.trunc() < f64::from(i32::MIN)
            || rate.trunc() > f64::from(i32::MAX)
        {
            return Err(PyValueError::new_err(
                "waiting rate is outside the supported integer range",
            ));
        }
        parsed.push(FrontArc {
            tail: front_window(required_field(arc, "tail_bounds")?)?,
            waiting: front_window(required_field(arc, "waiting_bounds")?)?,
            rate,
            base: prepared_machine_function(arc, "base")?,
            dependent: front_input(required_field(arc, "dependent")?.cast::<PyDict>()?)?,
            lower: front_input(required_field(arc, "dependent_lower")?.cast::<PyDict>()?)?,
            upper: front_input(required_field(arc, "dependent_upper")?.cast::<PyDict>()?)?,
        });
    }
    Ok(FrontProgram {
        source: front_window(required_field(dictionary, "source_bounds")?)?,
        terminal: front_window(required_field(dictionary, "terminal_bounds")?)?,
        root: front_input(required_field(dictionary, "root")?.cast::<PyDict>()?)?,
        arcs: parsed,
    })
}

#[pymethods]
impl EncodedFunction {
    /// Construct literal argument/value knots with a strict closed domain.
    #[staticmethod]
    fn knots(points: Vec<(f64, f64)>) -> PyResult<Self> {
        let function =
            PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            definition: FunctionDefinition::Knots { function },
        })
    }

    /// The first pair is (initial value, initial coordinate); each later
    /// pair is (segment slope, ending coordinate). These are not knot pairs.
    #[staticmethod]
    #[pyo3(signature = (prepared_pairs, policy="strict_domain"))]
    fn slope_segments(prepared_pairs: Vec<(f64, f64)>, policy: &str) -> PyResult<Self> {
        let policy = evaluation_policy(policy)?;
        let function = PreparedSlopeFunction::from_prepared_pairs(prepared_pairs)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            definition: FunctionDefinition::SlopeSegments { function, policy },
        })
    }

    /// Prepare an affine descriptor on a closed domain. Subsequent evaluation
    /// uses the anchored slope representation rather than knot interpolation.
    #[staticmethod]
    #[pyo3(signature = (intercept, slope, lower, upper, policy="strict_domain"))]
    fn affine(intercept: f64, slope: f64, lower: f64, upper: f64, policy: &str) -> PyResult<Self> {
        let policy = evaluation_policy(policy)?;
        let function = RawAffine::new(intercept, slope)
            .and_then(|affine| affine.prepare(lower, upper))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            definition: FunctionDefinition::SlopeSegments { function, policy },
        })
    }

    /// Return None outside the accepted domain; invalid arithmetic raises
    /// ValueError instead of becoming a feasible value or an empty function.
    fn value(&self, argument: f64) -> PyResult<Option<f64>> {
        self.definition
            .value(argument)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Export the tagged Rust/network wire definition without converting its
    /// arithmetic representation. Policy names use the public snake_case wire format.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let definition = PyDict::new(py);
        let function = PyDict::new(py);
        definition.set_item("encoding", self.encoding())?;
        match &self.definition {
            FunctionDefinition::Knots { function: knots } => {
                function.set_item("points", &knots.points)?;
            }
            FunctionDefinition::SlopeSegments {
                function: slopes,
                policy,
            } => {
                function.set_item("pairs", slopes.pairs().to_vec())?;
                definition.set_item(
                    "policy",
                    match policy {
                        EvaluationPolicy::StrictDomain => "strict_domain",
                        EvaluationPolicy::EndpointTolerance => "endpoint_tolerance",
                    },
                )?;
            }
        }
        definition.set_item("function", function)?;
        Ok(definition)
    }

    /// Validate a tagged Rust/network wire definition, including all nested
    /// fields, prepared-pair ordering and the explicit evaluation policy.
    #[staticmethod]
    fn from_dict(definition: &Bound<'_, PyDict>) -> PyResult<Self> {
        let encoding = required_field(definition, "encoding")?.extract::<String>()?;
        let function = required_field(definition, "function")?;
        let function = function.cast::<PyDict>()?;
        match encoding.as_str() {
            "knots" => {
                exact_fields(definition, &["encoding", "function"])?;
                exact_fields(function, &["points"])?;
                Self::knots(wire_pairs(&required_field(function, "points")?)?)
            }
            "slope_segments" => {
                exact_fields(definition, &["encoding", "function", "policy"])?;
                exact_fields(function, &["pairs"])?;
                let policy = required_field(definition, "policy")?.extract::<String>()?;
                Self::slope_segments(wire_pairs(&required_field(function, "pairs")?)?, &policy)
            }
            _ => Err(PyValueError::new_err("unknown function encoding")),
        }
    }

    #[getter]
    fn encoding(&self) -> &'static str {
        match &self.definition {
            FunctionDefinition::Knots { .. } => "knots",
            FunctionDefinition::SlopeSegments { .. } => "slope_segments",
        }
    }

    #[getter]
    fn policy(&self) -> Option<&'static str> {
        match &self.definition {
            FunctionDefinition::Knots { .. } => None,
            FunctionDefinition::SlopeSegments { policy, .. } => Some(match policy {
                EvaluationPolicy::StrictDomain => "strict_domain",
                EvaluationPolicy::EndpointTolerance => "endpoint_tolerance",
            }),
        }
    }
}

#[pyclass(module = "pybaldes._core")]
pub struct ResourceModel {
    resources: Vec<CoupledResource>,
    paired_waiting_base_resources: Vec<(usize, usize)>,
    paired_machine_inputs: Vec<(usize, Vec<PreparedMachineArc>)>,
    paired_machine_roots: Vec<(usize, PreparedSlopeFunction)>,
    paired_machine_terminal_bounds: Vec<(usize, baldes::graph::Window)>,
    paired_machine_waiting_rates: Vec<(usize, Vec<f64>)>,
    paired_front_programs: Vec<(usize, FrontProgram)>,
}
impl ResourceModel {
    fn push(&mut self, resource: CoupledResource) -> usize {
        let id = self.resources.len();
        self.resources.push(resource);
        id
    }
}
#[pymethods]
impl ResourceModel {
    #[new]
    fn new() -> Self {
        Self {
            resources: Vec::new(),
            paired_waiting_base_resources: Vec::new(),
            paired_machine_inputs: Vec::new(),
            paired_machine_roots: Vec::new(),
            paired_machine_terminal_bounds: Vec::new(),
            paired_machine_waiting_rates: Vec::new(),
            paired_front_programs: Vec::new(),
        }
    }
    /// Map a slope-paired resource to its physical scalar base resource for
    /// waiting windows. Pass None to remove the mapping. Changes take effect
    /// on the next install; the physical resource is validated then.
    fn set_paired_waiting_base_resource(
        &mut self,
        resource: usize,
        base_resource: Option<usize>,
    ) -> PyResult<()> {
        match self.resources.get(resource) {
            Some(CoupledResource::PairedFront { .. }) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "waiting base resource requires a paired resource",
                ))
            }
        }
        self.paired_waiting_base_resources
            .retain(|&(id, _)| id != resource);
        if let Some(base_resource) = base_resource {
            self.paired_waiting_base_resources
                .push((resource, base_resource));
        }
        Ok(())
    }
    /// Store exact prepared machine inputs in physical-arc insertion order.
    /// Pass None to remove them. Graph validation occurs on the next install;
    /// storing these inputs does not by itself enable machine pricing.
    fn set_paired_machine_inputs(
        &mut self,
        py: Python<'_>,
        resource: usize,
        arcs: Option<Vec<Py<PyDict>>>,
    ) -> PyResult<()> {
        match self.resources.get(resource) {
            Some(CoupledResource::PairedFront { .. }) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "machine inputs require a paired resource",
                ))
            }
        }
        let parsed = arcs
            .map(|values| {
                values
                    .iter()
                    .map(|value| prepared_machine_arc(value.bind(py)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        self.paired_machine_inputs.retain(|(id, _)| *id != resource);
        if let Some(parsed) = parsed {
            self.paired_machine_inputs.push((resource, parsed));
        }
        Ok(())
    }
    /// Store a prepared dependent vector for the machine root.
    fn set_paired_machine_root(
        &mut self,
        py: Python<'_>,
        resource: usize,
        root: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        if !matches!(
            self.resources.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return Err(PyValueError::new_err(
                "machine root requires a paired resource",
            ));
        }
        let parsed = root
            .map(|value| {
                let function = value.bind(py);
                exact_fields(function, &["pairs"])?;
                PreparedSlopeFunction::from_machine_pairs(wire_pairs(&required_field(
                    function, "pairs",
                )?)?)
                .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .transpose()?;
        self.paired_machine_roots.retain(|(id, _)| *id != resource);
        if let Some(parsed) = parsed {
            self.paired_machine_roots.push((resource, parsed));
        }
        Ok(())
    }
    /// Set the terminal base window used to prepare a backward machine root.
    /// Pass None to remove it; changes take effect on the next install.
    fn set_paired_machine_terminal_bounds(
        &mut self,
        resource: usize,
        bounds: Option<(f64, f64)>,
    ) -> PyResult<()> {
        let window = bounds
            .map(|(lower, upper)| {
                baldes::graph::Window::new(lower, upper)
                    .map_err(|error| PyValueError::new_err(error.message))
            })
            .transpose()?;
        self.paired_machine_terminal_bounds
            .retain(|(id, _)| *id != resource);
        if let Some(window) = window {
            self.paired_machine_terminal_bounds.push((resource, window));
        }
        Ok(())
    }

    /// Retain the original per-arc waiting rates for paired path recording.
    fn set_paired_machine_waiting_rates(
        &mut self,
        resource: usize,
        rates: Option<Vec<f64>>,
    ) -> PyResult<()> {
        if !matches!(
            self.resources.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return Err(PyValueError::new_err(
                "waiting rates require a paired resource",
            ));
        }
        if rates
            .as_ref()
            .is_some_and(|rates| rates.iter().any(|rate| !rate.is_finite()))
        {
            return Err(PyValueError::new_err("waiting rates must be finite"));
        }
        self.paired_machine_waiting_rates
            .retain(|(id, _)| *id != resource);
        if let Some(rates) = rates {
            self.paired_machine_waiting_rates.push((resource, rates));
        }
        Ok(())
    }

    /// Store the complete paired front program for preparation during install.
    fn set_paired_front_program(
        &mut self,
        py: Python<'_>,
        resource: usize,
        program: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        if !matches!(
            self.resources.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return Err(PyValueError::new_err(
                "front program requires a paired resource",
            ));
        }
        let parsed = program
            .map(|program| front_program(program.bind(py)))
            .transpose()?;
        self.paired_front_programs.retain(|(id, _)| *id != resource);
        if let Some(program) = parsed {
            self.paired_front_programs.push((resource, program));
        }
        Ok(())
    }

    /// Install resource kernels on this solver's physical graph in place.
    /// Existing options remain; labels, fixing and enumeration are invalidated.
    /// Paired resources use the prepared front program.
    #[pyo3(signature = (solver, *, front_program_domain=false))]
    fn install(&self, solver: &mut Solver, front_program_domain: bool) -> PyResult<()> {
        run(|| {
            let solver = solver.solver.get_mut().unwrap();
            // Validate all resource and window changes before publishing them.
            let mut checked = solver.graph.clone();
            checked.set_coupled_resources(&self.resources)?;
            for &(resource, base_resource) in &self.paired_waiting_base_resources {
                checked.set_paired_waiting_windows_from_resource(resource, base_resource)?;
            }
            for (resource, arcs) in &self.paired_machine_inputs {
                checked.set_paired_machine_inputs(*resource, Some(arcs))?;
            }
            for (resource, root) in &self.paired_machine_roots {
                checked.set_paired_machine_root(*resource, Some(root))?;
            }
            for &(resource, bounds) in &self.paired_machine_terminal_bounds {
                use baldes::graph::coupled::descriptor::{FrontierBudget, FrontierOutcome};
                match checked.install_paired_machine_backward_root(
                    resource,
                    bounds,
                    FrontierBudget::default(),
                ) {
                    FrontierOutcome::Complete(()) => {}
                    other => {
                        return Err(baldes::core::error::OrError {
                            kind: baldes::core::error::ErrorKind::InvalidArgument,
                            message: format!("paired machine backward root unavailable: {other:?}"),
                        })
                    }
                }
            }
            for (resource, program) in &self.paired_front_programs {
                use baldes::graph::coupled::{
                    descriptor::{FrontierBudget, FrontierOutcome},
                    MachineArcFunctionInput,
                };
                if self
                    .paired_machine_inputs
                    .iter()
                    .any(|(id, _)| id == resource)
                    || self
                        .paired_machine_roots
                        .iter()
                        .any(|(id, _)| id == resource)
                    || self
                        .paired_machine_terminal_bounds
                        .iter()
                        .any(|(id, _)| id == resource)
                    || self
                        .paired_machine_waiting_rates
                        .iter()
                        .any(|(id, _)| id == resource)
                {
                    return Err(baldes::core::error::OrError {
                        kind: baldes::core::error::ErrorKind::InvalidArgument,
                        message: "paired front program conflicts with separate inputs".into(),
                    });
                }
                let arcs: Vec<_> = program
                    .arcs
                    .iter()
                    .enumerate()
                    .map(|(physical_arc, arc)| MachineArcFunctionInput {
                        physical_arc,
                        tail_bounds: arc.tail,
                        waiting_bounds: arc.waiting,
                        waiting_rate: arc.rate.trunc() as i32,
                        base: &arc.base,
                        dependent: arc.dependent.input(),
                        dependent_lower: arc.lower.input(),
                        dependent_upper: arc.upper.input(),
                    })
                    .collect();
                let rates: Vec<_> = program.arcs.iter().map(|arc| arc.rate).collect();
                match checked.prepare_paired_front_program_with_rates(
                    *resource,
                    &arcs,
                    program.root.input(),
                    program.source,
                    program.terminal,
                    &rates,
                    FrontierBudget::default(),
                ) {
                    FrontierOutcome::Complete(()) => {}
                    other => {
                        return Err(baldes::core::error::OrError {
                            kind: baldes::core::error::ErrorKind::InvalidArgument,
                            message: format!("paired front program unavailable: {other:?}"),
                        })
                    }
                }
            }
            for (resource, rates) in &self.paired_machine_waiting_rates {
                checked.set_paired_machine_waiting_rates(*resource, rates)?;
            }
            if front_program_domain {
                checked.set_front_program_domain(true)?;
            }
            checked.validate()?;
            solver.graph = checked;
            solver.restore_arcs();
            Ok(())
        })
    }
    /// Arc vectors use original insertion order. Bits represent allowed values.
    /// ``vertices`` holds one mask per vertex; ``relaxed`` makes those masks
    /// checked before forgotten vertices reset the domain; the master tightens
    /// memory along improper routes.
    #[pyo3(signature = (initial, allowed, vertices=None, relaxed=false))]
    fn add_selection(
        &mut self,
        initial: u64,
        allowed: Vec<u64>,
        vertices: Option<Vec<u64>>,
        relaxed: bool,
    ) -> usize {
        self.push(CoupledResource::Selection {
            initial,
            allowed,
            vertices: vertices.unwrap_or_default(),
            relaxed,
        })
    }
    /// Deliver before picking up at each stop. Quantities belong to arc heads.
    /// Loads up to `capacity + tolerance` are feasible.
    #[pyo3(signature = (capacity, deliveries, pickups, tolerance=1e-6))]
    fn add_simultaneous_pickup_delivery(
        &mut self,
        capacity: f64,
        deliveries: Vec<f64>,
        pickups: Vec<f64>,
        tolerance: f64,
    ) -> usize {
        self.push(CoupledResource::SimultaneousPickupDelivery {
            capacity,
            deliveries,
            pickups,
            tolerance,
        })
    }
    /// x += increment, cost += x * weight on each arc. ``sink_upper`` and
    /// ``source_upper`` are this resource's upper bounds at the sink and the
    /// source; dominance uses them to trade a larger state against cost.
    #[pyo3(signature = (initial, increments, weights, sink_upper=None, source_upper=None))]
    fn add_cumulative_cost(
        &mut self,
        initial: f64,
        increments: Vec<f64>,
        weights: Vec<f64>,
        sink_upper: Option<f64>,
        source_upper: Option<f64>,
    ) -> usize {
        self.push(CoupledResource::CumulativeCost {
            initial,
            increments,
            weights,
            sink_upper,
            source_upper,
        })
    }
    /// Zero-initialized cumulative resource with independent dominance bounds.
    #[pyo3(signature = (increments, weights, forward_bound=None, backward_bound=None))]
    fn add_custom_resource(
        &mut self,
        increments: Vec<f64>,
        weights: Vec<f64>,
        forward_bound: Option<f64>,
        backward_bound: Option<f64>,
    ) -> usize {
        self.push(CoupledResource::CustomResource {
            increments,
            weights,
            forward_bound,
            backward_bound,
        })
    }
    /// Optimize one shared free initial value over all per-arc PWL charges.
    fn add_interval_cost(
        &mut self,
        initial_lower: f64,
        initial_upper: f64,
        lower: f64,
        upper: f64,
        increments: Vec<f64>,
        functions: Vec<Vec<(f64, f64)>>,
    ) -> PyResult<usize> {
        let initial = run(|| baldes::Window::new(initial_lower, initial_upper))?;
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        let functions = run(|| {
            functions
                .into_iter()
                .map(|points| {
                    PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
                })
                .collect::<baldes::Result<Vec<_>>>()
        })?;
        Ok(self.push(CoupledResource::IntervalCost {
            initial,
            bounds,
            increments,
            functions,
        }))
    }
    /// Optimize a shared initial interval using prepared slope-front operators.
    /// Functions must retain slope encoding and a matching leaf policy:
    /// strict_domain for strict_domain, endpoint_tolerance for front_algebra.
    /// The result is the computed operator-front minimum, not a global minimum
    /// of a sequential binary64 source program.
    #[pyo3(signature = (initial_lower, initial_upper, lower, upper, increments, functions, policy="strict_domain"))]
    fn add_slope_interval_cost(
        &mut self,
        initial_lower: f64,
        initial_upper: f64,
        lower: f64,
        upper: f64,
        increments: Vec<f64>,
        functions: Vec<PyRef<'_, EncodedFunction>>,
        policy: &str,
    ) -> PyResult<usize> {
        let (policy, leaf_policy) = front_policy(policy)?;
        let functions = functions
            .into_iter()
            .map(|function| slope_function(&function.definition, leaf_policy))
            .collect::<PyResult<Vec<_>>>()?;
        let initial = baldes::Window::new(initial_lower, initial_upper)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let bounds = baldes::Window::new(lower, upper)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if increments.len() != functions.len()
            || increments.iter().any(|value| !value.is_finite())
            || ![initial_lower, initial_upper, lower, upper]
                .iter()
                .all(|value| value.is_finite())
            || initial_lower < lower
            || initial_upper > upper
        {
            return Err(PyValueError::new_err("slope interval cost requires finite nested windows, finite increments, and matching function/increment counts"));
        }
        Ok(self.push(CoupledResource::SlopeIntervalCost {
            initial,
            bounds,
            increments,
            functions,
            policy,
        }))
    }
    /// Add a paired resource whose functions are supplied by a front program.
    #[pyo3(signature = (mode="feasibility"))]
    fn add_paired_front(&mut self, mode: &str) -> PyResult<usize> {
        let mode = match mode {
            "feasibility" => PairedResourceMode::Feasibility,
            "cost_unbounded" => PairedResourceMode::CostUnbounded,
            "cost_bounded" => PairedResourceMode::CostBounded,
            _ => return Err(PyValueError::new_err("unknown paired resource mode")),
        };
        Ok(self.push(CoupledResource::PairedFront { mode }))
    }
    /// Replace a numeric state with its per-arc piecewise-linear map.
    fn add_piecewise_linear_resource(
        &mut self,
        initial: f64,
        lower: f64,
        upper: f64,
        functions: Vec<Vec<(f64, f64)>>,
    ) -> PyResult<usize> {
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        let functions = run(|| {
            functions
                .into_iter()
                .map(|points| {
                    PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
                })
                .collect::<baldes::Result<Vec<_>>>()
        })?;
        Ok(self.push(CoupledResource::PiecewiseLinearResource {
            initial,
            bounds,
            functions,
        }))
    }
    /// Charge a function of the earlier parent's updated state on every arc.
    fn add_piecewise_linear_cost(
        &mut self,
        parent: usize,
        functions: Vec<Vec<(f64, f64)>>,
    ) -> PyResult<usize> {
        let functions = run(|| {
            functions
                .into_iter()
                .map(|points| {
                    PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
                })
                .collect::<baldes::Result<Vec<_>>>()
        })?;
        Ok(self.push(CoupledResource::PiecewiseLinearCost { parent, functions }))
    }
    /// y += a continuous piecewise-linear function of the updated parent.
    fn add_piecewise_linear_dependent(
        &mut self,
        parent: usize,
        initial: f64,
        lower: f64,
        upper: f64,
        functions: Vec<Vec<(f64, f64)>>,
    ) -> PyResult<usize> {
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        let functions = run(|| {
            functions
                .into_iter()
                .map(|points| {
                    PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
                })
                .collect::<baldes::Result<Vec<_>>>()
        })?;
        Ok(self.push(CoupledResource::PiecewiseLinearDependent {
            parent,
            initial,
            bounds,
            functions,
        }))
    }
    /// Replace a numeric state using explicit per-arc function encodings.
    fn add_encoded_piecewise_linear_resource(
        &mut self,
        initial: f64,
        lower: f64,
        upper: f64,
        functions: Vec<PyRef<'_, EncodedFunction>>,
    ) -> PyResult<usize> {
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        Ok(self.push(CoupledResource::EncodedPiecewiseLinearResource {
            initial,
            bounds,
            functions: functions.iter().map(|f| f.definition.clone()).collect(),
        }))
    }

    /// y += f(parent_after), preserving each function's encoding and policy.
    fn add_encoded_piecewise_linear_dependent(
        &mut self,
        parent: usize,
        initial: f64,
        lower: f64,
        upper: f64,
        functions: Vec<PyRef<'_, EncodedFunction>>,
    ) -> PyResult<usize> {
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        Ok(self.push(CoupledResource::EncodedPiecewiseLinearDependent {
            parent,
            initial,
            bounds,
            functions: functions.iter().map(|f| f.definition.clone()).collect(),
        }))
    }

    /// Charge an explicitly encoded function of the updated parent state.
    fn add_encoded_piecewise_linear_cost(
        &mut self,
        parent: usize,
        functions: Vec<PyRef<'_, EncodedFunction>>,
    ) -> usize {
        self.push(CoupledResource::EncodedPiecewiseLinearCost {
            parent,
            functions: functions.iter().map(|f| f.definition.clone()).collect(),
        })
    }
    /// y += offset + factor * parent's updated value, with strict bounds.
    fn add_dependent(
        &mut self,
        parent: usize,
        initial: f64,
        lower: f64,
        upper: f64,
        offsets: Vec<f64>,
        factors: Vec<f64>,
    ) -> PyResult<usize> {
        let bounds = run(|| baldes::Window::new(lower, upper))?;
        Ok(self.push(CoupledResource::Dependent {
            parent,
            initial,
            bounds,
            offsets,
            factors,
        }))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EncodedFunction>()?;
    m.add_class::<ResourceModel>()?;
    Ok(())
}
