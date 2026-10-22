//! Restricted-master dual requests; these never certify an external domain.
use super::*;
pub use certificate::ImportCertificate;
pub(super) mod certificate;
mod enclosure;
mod envelope;
mod feasibility;
mod objective;
mod oracle;
pub use oracle::{ImportOracleReply, ImportPricingMinimum};

/// Which objective the numerical pricing point represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportPricingKind {
    Objective,
    Feasibility,
}

/// Numerical objective for an external pricing subproblem:
/// per_column + sum(coordinate_price * vector_entry), plus additional
/// column cost in Objective mode. Feasibility mode ignores all original costs.
#[derive(Clone, Debug)]
pub struct ImportCosts {
    pub kind: ImportPricingKind,
    pub per_column: f64,
    pub coordinates: Vec<(i32, f64)>,
}

/// A numerical dual point for generating new columns. No full-domain bound,
/// infeasibility conclusion or completion flag is carried by this object.
#[derive(Clone, Debug)]
pub struct ImportPricing<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    epoch: Rc<()>,
    context: Rc<Context<D>>,
    definition: Rc<Definition<D>>,
    position: usize,
    duals: Vec<f64>,
    shared_offset: usize,
    shared_rows: usize,
    kind: ImportPricingKind,
    violation: Option<f64>,
    routing: Vec<NetworkRoute>,
    retained_bound: Option<f64>,
    open_groups: BTreeSet<i32>,
}
impl<const D: usize> ImportPricing<D> {
    pub(in crate::master::multi_pricing::imported) fn open_group_ids(&self) -> &BTreeSet<i32> {
        &self.open_groups
    }
    pub fn kind(&self) -> ImportPricingKind {
        self.kind
    }
    /// Numerical artificial activity in the restricted phase-I master only.
    pub fn restricted_violation(&self) -> Option<f64> {
        self.violation
    }

    pub fn subproblem(&self) -> &Subproblem {
        &self.definition.subproblems[self.position]
    }

    /// Export the numerical linear objective needed by an external optimizer.
    /// All base and explicit group multiplicity rows enter the constant term.
    pub fn costs(&self) -> Result<ImportCosts> {
        let group = self.subproblem();
        let used: HashSet<_> = group.columns.iter().map(|c| c.id).collect();
        let id = (0..=i32::MAX)
            .find(|id| !used.contains(id))
            .ok_or_else(|| invalid("pricing candidate IDs exhausted"))?;
        let per_column = self.reduced_cost(Column {
            id,
            cost: 0.,
            bounds: Bounds {
                lower: 0.,
                upper: f64::INFINITY,
            },
            integer: false,
            entries: vec![],
        })?;
        let compiled = lifecycle::compile_definition(&self.definition, Some(&self.context))?;
        let coordinates = group
            .variables
            .iter()
            .map(|id| {
                (
                    *id,
                    self.duals[self.shared_offset + compiled.registry.links[id]],
                )
            })
            .collect();
        Ok(ImportCosts {
            kind: self.kind,
            per_column,
            coordinates,
        })
    }

    /// Numerical reduced cost of a new, zero-lower-bound column in this group.
    /// Recompile identity links and every explicit multiplicity restriction so
    /// shifted internal column IDs cannot retarget the candidate. Additional
    /// original-variable objective costs enter through the coordinate duals.
    /// This value is a search signal, not a certified global pricing bound.
    pub fn reduced_cost(&self, column: Column) -> Result<f64> {
        let variable = self.candidate(column)?;
        let mut reduced = if self.kind == ImportPricingKind::Objective {
            variable.cost
        } else {
            0.
        };
        for &(row, a) in &variable.entries {
            let dual = self
                .duals
                .get(self.shared_offset + row)
                .ok_or_else(|| invalid("restricted import dual row mapping"))?;
            reduced = (-a).mul_add(*dual, reduced);
        }
        if !reduced.is_finite() {
            return Err(invalid("import reduced cost overflow"));
        }
        Ok(reduced)
    }
    fn candidate(&self, column: Column) -> Result<MasterVariable> {
        self.candidate_at(self.position, column)
    }
    fn candidate_at(&self, position: usize, column: Column) -> Result<MasterVariable> {
        if column.bounds.lower != 0. {
            return Err(invalid(
                "pricing candidate requires zero quantity lower bound",
            ));
        }
        let id = column.id;
        let group = self.definition.subproblems[position].id;
        let old = &self.definition;
        let mut definition = Definition {
            problem: old.problem.clone(),
            variables: old.variables.clone(),
            subproblems: old.subproblems.clone(),
            envelopes: old.envelopes.clone(),
        };
        // Algebra queries may use symbolic vectors outside the permitted box
        // (e.g. the objective intercept). Acceptance enforces the envelope.
        definition.envelopes.remove(&group);
        definition.subproblems[position].columns.push(column);
        let model = lifecycle::compile_definition(&definition, Some(&self.context))?;
        if model.problem.rows.len() != self.shared_rows {
            return Err(invalid("pricing candidate changed original row domain"));
        }
        let variable_id = model.registry.columns[&(group, id)];
        let variable = model
            .variables
            .into_iter()
            .find(|v| v.id == variable_id)
            .unwrap();
        Ok(variable)
    }
}

/// Shared numerical request selection for collection drivers.
pub(super) fn next_point<const D: usize>(
    session: &ColumnImport<D>,
    options: &Options,
    deadline: Instant,
) -> Result<Option<ImportPricing<D>>> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Ok(None);
    }
    let mut phase_options = options.clone();
    phase_options.cg.time_limit = phase_options.cg.time_limit.min(remaining);
    let phase = match session.feasibility_pricing(phase_options) {
        Ok(point) => point,
        Err(_) if Instant::now() >= deadline => return Ok(None),
        Err(error) => return Err(error),
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Ok(None);
    }
    if phase.restricted_violation().unwrap_or(f64::INFINITY) > 0. {
        return Ok(Some(phase));
    }
    let mut objective_options = options.clone();
    objective_options.cg.time_limit = objective_options.cg.time_limit.min(remaining);
    match session.pricing(objective_options) {
        Ok(point) => Ok(Some(point)),
        Err(_) if Instant::now() >= deadline => Ok(None),
        Err(error) => Err(error),
    }
}
