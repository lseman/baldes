//! Explicit original-vector domain envelopes, retained across rebuilds.
use super::*;

/// Every column in this group must have additional cost at least cost_lower
/// and original coordinates inside the declared intervals. Omitted entries
/// are zero. Every coordinate in the group's support must be declared once.
#[derive(Clone, Debug)]
pub struct ColumnEnvelope {
    pub cost_lower: f64,
    pub coordinates: Vec<(i32, Bounds)>,
}
impl ColumnEnvelope {
    fn check(&self, group: &Subproblem) -> Result<()> {
        if self.cost_lower.is_nan() || self.cost_lower == f64::INFINITY {
            return Err(invalid("column envelope cost lower bound"));
        }
        let mut seen = HashSet::new();
        if self.coordinates.len() != group.variables.len()
            || self.coordinates.iter().any(|(id, b)| {
                !group.variables.contains(id)
                    || !seen.insert(*id)
                    || !super::super::master_variables::valid_bounds(*b)
            })
        {
            return Err(invalid("column envelope coordinate support or bounds"));
        }
        for column in &group.columns {
            if column.cost < self.cost_lower {
                return Err(invalid("column violates envelope cost"));
            }
            for &(id, bounds) in &self.coordinates {
                let value = column
                    .entries
                    .iter()
                    .find(|(key, _)| *key == id)
                    .map_or(0., |(_, a)| *a);
                if value < bounds.lower || value > bounds.upper {
                    return Err(invalid("column violates envelope coordinates"));
                }
            }
        }
        Ok(())
    }
}
pub(super) fn validate<const D: usize>(definition: &Definition<D>) -> Result<()> {
    for (id, envelope) in &definition.envelopes {
        let group = definition
            .subproblems
            .iter()
            .find(|g| g.id == *id)
            .ok_or_else(|| invalid("unknown envelope subproblem"))?;
        envelope.check(group)?;
    }
    Ok(())
}
impl<const D: usize> ColumnImport<D> {
    /// Restrict the permitted domain for this group, including future columns.
    /// Repeated declarations intersect; invalid declarations preserve the input.
    pub fn with_envelope(&self, mut envelope: ColumnEnvelope) -> Result<Self> {
        envelope.check(self.subproblem())?;
        if let Some(previous) = self.definition.envelopes.get(&self.subproblem().id) {
            envelope.cost_lower = envelope.cost_lower.max(previous.cost_lower);
            for (id, bounds) in &mut envelope.coordinates {
                let old = previous
                    .coordinates
                    .iter()
                    .find(|(key, _)| key == id)
                    .unwrap()
                    .1;
                bounds.lower = bounds.lower.max(old.lower);
                bounds.upper = bounds.upper.min(old.upper);
            }
            envelope.check(self.subproblem())?;
        }
        let old = &self.definition;
        let mut definition = Definition {
            problem: old.problem.clone(),
            variables: old.variables.clone(),
            subproblems: old.subproblems.clone(),
            envelopes: old.envelopes.clone(),
        };
        definition.envelopes.insert(self.subproblem().id, envelope);
        let mut session = self.clone();
        session.definition = Rc::new(definition);
        Ok(session)
    }
    pub fn envelope(&self) -> Option<&ColumnEnvelope> {
        self.definition.envelopes.get(&self.subproblem().id)
    }
}
