//! Direct coupled-resource model installation for the low-level Python API.
use super::{run, Solver};
use baldes::graph::coupled::{
    descriptor::{PairedCostNormalization, RawAffine, SlopeRelationArc},
    CoupledResource, EncodedFunction as FunctionDefinition, EvaluationPolicy, FrontOperationPolicy,
    PairedResourceArc, PairedResourceMode, PiecewiseLinearFunction, PreparedSlopeFunction,
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
        "native_tolerance" => Ok(EvaluationPolicy::NativeTolerance),
        _ => Err(PyValueError::new_err(
            "policy must be 'strict_domain' or 'native_tolerance'",
        )),
    }
}

fn front_policy(policy: &str) -> PyResult<(FrontOperationPolicy, EvaluationPolicy)> {
    match policy {
        "strict_domain" => Ok((
            FrontOperationPolicy::StrictDomain,
            EvaluationPolicy::StrictDomain,
        )),
        "native_front_algebra" => Ok((
            FrontOperationPolicy::NativeFrontAlgebra,
            EvaluationPolicy::NativeTolerance,
        )),
        _ => Err(PyValueError::new_err(
            "front policy must be 'strict_domain' or 'native_front_algebra'",
        )),
    }
}

fn slope_function(
    definition: &FunctionDefinition,
    expected: EvaluationPolicy,
) -> PyResult<PreparedSlopeFunction> {
    match definition {
        FunctionDefinition::SlopeSegments { function, policy } if *policy == expected => Ok(function.clone()),
        FunctionDefinition::SlopeSegments { .. } => Err(PyValueError::new_err("slope function evaluation policy must match front policy: strict_domain requires strict_domain; native_front_algebra requires native_tolerance")),
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
                        EvaluationPolicy::NativeTolerance => "native_tolerance",
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
                EvaluationPolicy::NativeTolerance => "native_tolerance",
            }),
        }
    }
}

#[pyclass(module = "pybaldes._core")]
pub struct ResourceModel {
    resources: Vec<CoupledResource>,
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
        }
    }
    /// Install resource kernels on this solver's physical graph in place.
    /// Existing options remain; labels, fixing and enumeration are invalidated.
    /// native_front_program_domain selects a computed native-program model;
    /// the default keeps source-control feasibility.
    #[pyo3(signature = (solver, *, native_front_program_domain=false))]
    fn install(&self, solver: &mut Solver, native_front_program_domain: bool) -> PyResult<()> {
        run(|| {
            let solver = solver.solver.get_mut().unwrap();
            if native_front_program_domain {
                // Validate the modelling choice before replacing live resources.
                let mut checked = solver.graph.clone();
                checked.set_coupled_resources(&self.resources)?;
                checked.set_native_front_program_domain(true)?;
            }
            solver.set_coupled_resources(&self.resources)?;
            if native_front_program_domain {
                solver.graph.set_native_front_program_domain(true)?;
            }
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
    /// strict_domain for strict_domain, native_tolerance for native_front_algebra.
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
    /// Retain a paired base/dependent relation with optional waiting.
    /// Arrival functions give the total base arrival, not its increment.
    /// Cost modes charge final dependent minimum minus initial minimum.
    #[pyo3(signature = (initial_lower, initial_upper, initial, arrivals, increments, lowers, uppers, waiting, mode="feasibility"))]
    fn add_paired_piecewise_linear(
        &mut self,
        initial_lower: f64,
        initial_upper: f64,
        initial: Vec<(f64, f64)>,
        arrivals: Vec<Vec<(f64, f64)>>,
        increments: Vec<Vec<(f64, f64)>>,
        lowers: Vec<Vec<(f64, f64)>>,
        uppers: Vec<Vec<(f64, f64)>>,
        waiting: Vec<i32>,
        mode: &str,
    ) -> PyResult<usize> {
        let resource = run(|| {
            let invalid = |message: &str| baldes::core::error::OrError {
                kind: baldes::core::error::ErrorKind::InvalidArgument,
                message: message.into(),
            };
            let function = |points: Vec<(f64, f64)>| {
                PiecewiseLinearFunction::new(points.into_iter().map(|(x, y)| [x, y]).collect())
            };
            let count = arrivals.len();
            if [increments.len(), lowers.len(), uppers.len(), waiting.len()]
                .iter()
                .any(|&n| n != count)
            {
                return Err(invalid("paired resource arc vector lengths differ"));
            }
            let mode = match mode {
                "feasibility" => PairedResourceMode::Feasibility,
                "cost_unbounded" => PairedResourceMode::CostUnbounded,
                "cost_bounded" => PairedResourceMode::CostBounded,
                _ => return Err(invalid("unknown paired resource mode")),
            };
            let arcs = arrivals
                .into_iter()
                .zip(increments)
                .zip(lowers)
                .zip(uppers)
                .zip(waiting)
                .map(|((((arrival, increment), lower), upper), waiting)| {
                    Ok(PairedResourceArc {
                        arrival: function(arrival)?,
                        increment: function(increment)?,
                        lower: function(lower)?,
                        upper: function(upper)?,
                        waiting,
                    })
                })
                .collect::<baldes::Result<Vec<_>>>()?;
            Ok(CoupledResource::PairedPiecewiseLinear {
                initial: function(initial)?,
                initial_base: baldes::Window::new(initial_lower, initial_upper)?,
                arcs,
                mode,
            })
        })?;
        Ok(self.push(resource))
    }
    /// Retain a paired slope-front relation with an explicit objective baseline.
    /// Only slope encodings with policies matching the chosen front operator
    /// are accepted. Cost modes default to charging the absolute final value.
    /// Native operator compatibility is distinct from source-control replay.
    #[pyo3(signature = (initial_lower, initial_upper, initial, arrivals, increments, lowers, uppers, waiting, mode="feasibility", policy="strict_domain", normalization="absolute_final_value"))]
    fn add_slope_paired(
        &mut self,
        initial_lower: f64,
        initial_upper: f64,
        initial: PyRef<'_, EncodedFunction>,
        arrivals: Vec<PyRef<'_, EncodedFunction>>,
        increments: Vec<PyRef<'_, EncodedFunction>>,
        lowers: Vec<PyRef<'_, EncodedFunction>>,
        uppers: Vec<PyRef<'_, EncodedFunction>>,
        waiting: Vec<i32>,
        mode: &str,
        policy: &str,
        normalization: &str,
    ) -> PyResult<usize> {
        let mode = match mode {
            "feasibility" => PairedResourceMode::Feasibility,
            "cost_unbounded" => PairedResourceMode::CostUnbounded,
            "cost_bounded" => PairedResourceMode::CostBounded,
            _ => return Err(PyValueError::new_err("unknown paired resource mode")),
        };
        let normalization =
            match normalization {
                "absolute_final_value" => PairedCostNormalization::AbsoluteFinalValue,
                "final_minus_initial_minimum" => PairedCostNormalization::FinalMinusInitialMinimum,
                _ => return Err(PyValueError::new_err(
                    "normalization must be 'absolute_final_value' or 'final_minus_initial_minimum'",
                )),
            };
        let (policy, leaf_policy) = front_policy(policy)?;
        let initial_base = baldes::Window::new(initial_lower, initial_upper)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if !initial_lower.is_finite() || !initial_upper.is_finite() {
            return Err(PyValueError::new_err(
                "slope paired initial window must be finite",
            ));
        }
        let initial = slope_function(&initial.definition, leaf_policy)?;
        let count = arrivals.len();
        if [increments.len(), lowers.len(), uppers.len(), waiting.len()]
            .iter()
            .any(|&n| n != count)
        {
            return Err(PyValueError::new_err(
                "paired resource arc vector lengths differ",
            ));
        }
        let arcs = arrivals
            .into_iter()
            .zip(increments)
            .zip(lowers)
            .zip(uppers)
            .zip(waiting)
            .map(|((((arrival, increment), lower), upper), waiting)| {
                Ok(SlopeRelationArc {
                    arrival: slope_function(&arrival.definition, leaf_policy)?,
                    increment: slope_function(&increment.definition, leaf_policy)?,
                    lower: slope_function(&lower.definition, leaf_policy)?,
                    upper: slope_function(&upper.definition, leaf_policy)?,
                    waiting,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.push(CoupledResource::SlopePaired {
            initial,
            initial_base,
            arcs,
            mode,
            normalization,
            policy,
        }))
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
