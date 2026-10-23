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
    // Compiled with this request's definition and restriction journal. Sharing
    // it across group views preserves row identities without recompilation.
    registry: Rc<Registry>,
    position: usize,
    duals: Vec<f64>,
    shared_offset: usize,
    kind: ImportPricingKind,
    violation: Option<f64>,
    routing: Vec<NetworkRoute>,
    retained_bound: Option<f64>,
    open_groups: BTreeSet<i32>,
}
impl<const D: usize> ImportPricing<D> {
    pub(in crate::master::mp::imported) fn open_group_ids(&self) -> &BTreeSet<i32> {
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
        let rows = &self.registry.group_rows[&group.id];
        let id = rows
            .next_column
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
        let coordinates = group
            .variables
            .iter()
            .map(|id| (*id, self.duals[self.shared_offset + rows.links[id]]))
            .collect();
        Ok(ImportCosts {
            kind: self.kind,
            per_column,
            coordinates,
        })
    }

    /// Numerical reduced cost of a new, zero-lower-bound column in this group.
    /// Use this request's compiled links and multiplicity restrictions so
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
    fn candidate(&self, column: Column) -> Result<Candidate> {
        self.candidate_at(self.position, column)
    }
    fn candidate_at(&self, position: usize, column: Column) -> Result<Candidate> {
        if column.bounds.lower != 0. {
            return Err(invalid(
                "pricing candidate requires zero quantity lower bound",
            ));
        }
        let group = self.definition.subproblems[position].id;
        if column.id < 0
            || self.registry.columns.contains_key(&(group, column.id))
            || !column.cost.is_finite()
        {
            return Err(invalid("imported column identity or cost"));
        }
        quantity_bounds(column.bounds)?;
        let rows = &self.registry.group_rows[&group];
        let mut entries = Vec::with_capacity(column.entries.len() + rows.multiplicity.len());
        for (id, a) in column.entries {
            let row = rows
                .links
                .get(&id)
                .filter(|_| a.is_finite())
                .ok_or_else(|| invalid("coalesced imported column entries required"))?;
            entries.push((*row, -a));
        }
        entries.sort_unstable_by_key(|&(row, _)| row);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("coalesced imported column entries required"));
        }
        entries.retain(|&(_, a)| a != 0.);
        // Link rows precede base multiplicity rows; journal multiplicity rows
        // follow in replay order. Keep the compiler's floating-point order.
        entries.extend(rows.multiplicity.iter().map(|&row| (row, 1.)));
        // Algebra queries may lie outside the envelope (e.g. its intercept).
        // Acceptance, rather than reduced-cost evaluation, enforces that box.
        Ok(Candidate {
            cost: column.cost,
            entries,
        })
    }
}

struct Candidate {
    cost: f64,
    entries: Vec<(usize, f64)>,
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

#[cfg(test)]
#[path = "../../../../../tests/unit/master/mp/imported/pricing/candidate_tests.rs"]
mod candidate_tests;
