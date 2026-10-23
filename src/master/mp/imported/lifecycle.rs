//! Rebuild wider declared domains from original input and explicit restrictions.
use super::*;

pub(super) struct Definition<const D: usize> {
    pub problem: Problem<D>,
    pub variables: Vec<MasterVariable>,
    pub subproblems: Vec<Subproblem>,
    pub envelopes: BTreeMap<i32, ColumnEnvelope>,
}
impl<const D: usize> std::fmt::Debug for Definition<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportDefinition")
            .field("shared_rows", &self.problem.rows.len())
            .field("routing_networks", &self.problem.networks.len())
            .field("original_variables", &self.variables.len())
            .field("imported_subproblems", &self.subproblems.len())
            .field("envelopes", &self.envelopes.len())
            .finish()
    }
}
#[derive(Clone, Debug)]
pub(super) enum Restriction {
    NonlinearRow {
        row: Row,
        kernels: nonlinear::RowKernels,
        original: Vec<(i32, f64)>,
    },
    Variable(i32, Bounds),
    Column(i32, i32, Bounds),
    Multiplicity(i32, Bounds),
    RoutingMultiplicity(i32, Bounds),
    FixedRoute(FixedRoute),
    SharedRow {
        row: Row,
        routing: Vec<RowCoefficients>,
        original: Vec<(i32, f64)>,
    },
}
#[derive(Debug)]
struct Step {
    restriction: Restriction,
    previous: Option<Rc<Step>>,
}
#[derive(Debug)]
pub(super) struct Context<const D: usize> {
    pub registry: Rc<Registry>,
    pub definition: Rc<Definition<D>>,
    restrictions: Option<Rc<Step>>,
}
impl<const D: usize> Context<D> {
    pub fn restricted(&self, restriction: Restriction) -> Rc<Self> {
        Rc::new(Self {
            registry: self.registry.clone(),
            definition: self.definition.clone(),
            restrictions: Some(Rc::new(Step {
                restriction,
                previous: self.restrictions.clone(),
            })),
        })
    }
    pub fn multiplicity_upper(&self, group: i32, initial: f64) -> f64 {
        self.ordered()
            .into_iter()
            .fold(initial, |upper, restriction| match restriction {
                Restriction::Multiplicity(id, bounds) if *id == group => upper.min(bounds.upper),
                _ => upper,
            })
    }
    fn ordered(&self) -> Vec<&Restriction> {
        let mut result = vec![];
        let mut next = self.restrictions.as_deref();
        while let Some(step) = next {
            result.push(&step.restriction);
            next = step.previous.as_deref();
        }
        result.reverse();
        result
    }
}

/// Reconstruct only model input and user restrictions. Derived finite-domain
/// bounds, loaded routing columns, certificates and smoothing are not valid
/// assumptions for a wider import domain and are deliberately not copied.
pub(super) fn rebuild<const D: usize>(
    definition: Rc<Definition<D>>,
    previous: Option<&Context<D>>,
    options: Options,
) -> Result<ImportedOutput<D>> {
    let Compiled {
        problem,
        variables,
        registry,
        fixed,
        nonlinear,
    } = compile_definition(&definition, previous)?;
    let context = Rc::new(Context {
        registry: Rc::new(registry),
        definition,
        restrictions: previous.and_then(|p| p.restrictions.clone()),
    });
    ImportedOutput::new(
        solve_with_fixed_routes(problem, variables, options, &fixed, &nonlinear)?,
        context,
    )
}

pub(super) struct Compiled<const D: usize> {
    pub problem: Problem<D>,
    pub variables: Vec<MasterVariable>,
    pub registry: Registry,
    pub fixed: Vec<FixedRoute>,
    pub nonlinear: Vec<(usize, nonlinear::RowKernels)>,
}

pub(super) fn compile_definition<const D: usize>(
    definition: &Definition<D>,
    previous: Option<&Context<D>>,
) -> Result<Compiled<D>> {
    super::envelope::validate(definition)?;
    let (mut problem, mut variables, registry) = compile::model(
        definition.problem.clone(),
        definition.variables.clone(),
        definition.subproblems.clone(),
    )?;
    let indices: HashMap<_, _> = variables
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id, i))
        .collect();
    let mut fixed = vec![];
    let mut nonlinear = vec![];
    if let Some(previous) = previous {
        for restriction in previous.ordered() {
            if let Restriction::NonlinearRow {
                row,
                kernels,
                original,
            } = restriction
            {
                kernels.validate(&problem, row)?;
                let index = problem.rows.len();
                problem.rows.push(row.clone());
                for &(id, a) in original {
                    if a != 0. {
                        variables[indices[&id]].entries.push((index, a));
                    }
                }
                nonlinear.push((index, kernels.clone()));
                continue;
            }
            if let Restriction::SharedRow {
                row,
                routing,
                original,
            } = restriction
            {
                let index = problem.rows.len();
                problem.rows.push(row.clone());
                for &(id, a) in original {
                    if a != 0. {
                        variables[indices[&id]].entries.push((index, a));
                    }
                }
                for coefficients in routing {
                    let network = problem
                        .networks
                        .iter_mut()
                        .find(|n| n.id == coefficients.network)
                        .ok_or_else(|| invalid("unknown shared-row routing subproblem"))?;
                    if coefficients.per_route != 0. {
                        network.route_entries.push((index, coefficients.per_route));
                    }
                    for (entries, &a) in network.arc_entries.iter_mut().zip(&coefficients.arcs) {
                        if a != 0. {
                            entries.push((index, a));
                        }
                    }
                }
                continue;
            }
            if let Restriction::FixedRoute(route) = restriction {
                fixed.push(route.clone());
                continue;
            }
            if let Restriction::RoutingMultiplicity(id, bounds) = *restriction {
                let network = problem
                    .networks
                    .iter_mut()
                    .find(|n| n.id == id)
                    .ok_or_else(|| invalid("unknown routing subproblem"))?;
                for (sense, rhs) in [
                    (RowSense::GreaterEqual, bounds.lower),
                    (RowSense::LessEqual, bounds.upper),
                ] {
                    if rhs.is_finite() {
                        let row = problem.rows.len();
                        problem.rows.push(Row { sense, rhs });
                        network.route_entries.push((row, 1.));
                    }
                }
                continue;
            }
            let (ids, bounds) = match *restriction {
                Restriction::RoutingMultiplicity(..)
                | Restriction::FixedRoute(..)
                | Restriction::SharedRow { .. }
                | Restriction::NonlinearRow { .. } => {
                    unreachable!()
                }
                Restriction::Variable(id, bounds) => (vec![id], bounds),
                Restriction::Column(group, column, bounds) => {
                    (vec![registry.columns[&(group, column)]], bounds)
                }
                Restriction::Multiplicity(group, bounds) => {
                    (registry.groups[&group].clone(), bounds)
                }
            };
            for (sense, rhs) in [
                (RowSense::GreaterEqual, bounds.lower),
                (RowSense::LessEqual, bounds.upper),
            ] {
                if !rhs.is_finite() {
                    continue;
                }
                let row = problem.rows.len();
                problem.rows.push(Row { sense, rhs });
                for id in &ids {
                    variables[indices[id]].entries.push((row, 1.));
                }
            }
        }
    }
    Ok(Compiled {
        problem,
        variables,
        registry,
        fixed,
        nonlinear,
    })
}

impl<const D: usize> ImportedSnapshot<D> {
    /// Solve a newly declared domain with additional columns in an existing
    /// imported group. Explicit original/column/multiplicity restrictions
    /// survive. Old derived bounds, pricing caches and publication bounds do
    /// not: adding columns can invalidate each of them. This rebuild is a
    /// fresh solve, not a dynamic pricing-completeness certificate.
    pub fn solve_with_columns(
        &self,
        subproblem: i32,
        columns: Vec<Column>,
        options: Options,
    ) -> Result<ImportedOutput<D>> {
        let position = self
            .context
            .definition
            .subproblems
            .iter()
            .position(|s| s.id == subproblem)
            .ok_or_else(|| invalid("unknown imported subproblem"))?;
        if columns.is_empty() {
            return self.solve(options);
        }
        let old = &self.context.definition;
        let mut definition = Definition {
            problem: old.problem.clone(),
            variables: old.variables.clone(),
            subproblems: old.subproblems.clone(),
            envelopes: old.envelopes.clone(),
        };
        definition.subproblems[position].columns.extend(columns);
        rebuild(Rc::new(definition), Some(&self.context), options)
    }
}
