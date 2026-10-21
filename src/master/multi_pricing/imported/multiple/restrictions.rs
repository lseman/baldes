//! Explicit child restrictions survive every subsequent source batch.
use super::*;

impl<const D: usize> MultiImport<D> {
    fn restricted(&self, restriction: Restriction) -> Result<Self> {
        let mut child = self.clone();
        child.context = self.context.restricted(restriction);
        let compiled = lifecycle::compile_definition(&child.definition, Some(&child.context))?;
        compiled.problem.validate()?;
        super::super::super::master_variables::validate(&compiled.problem, &compiled.variables)?;
        Ok(child)
    }
    /// Restrict an original coordinate, including contributions of future columns.
    pub fn with_variable_bounds(&self, id: i32, bounds: Bounds) -> Result<Self> {
        if !self.definition.variables.iter().any(|v| v.id == id)
            || !super::super::super::master_variables::valid_bounds(bounds)
        {
            return Err(invalid("import original variable or bounds"));
        }
        self.restricted(Restriction::Variable(id, bounds))
    }
    /// Restrict a known column by its stable, group-qualified identity.
    pub fn with_column_bounds(&self, group: i32, column: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        if !self
            .definition
            .subproblems
            .iter()
            .any(|g| g.id == group && g.columns.iter().any(|c| c.id == column))
        {
            return Err(invalid("unknown imported column"));
        }
        self.restricted(Restriction::Column(group, column, bounds))
    }
    /// Total group quantity, with coefficients on accepted and future columns.
    pub fn with_multiplicity(&self, group: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        if !self.definition.subproblems.iter().any(|g| g.id == group) {
            return Err(invalid("unknown imported subproblem"));
        }
        self.restricted(Restriction::Multiplicity(group, bounds))
    }
    pub fn with_routing_multiplicity(&self, network: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        if !self
            .definition
            .problem
            .networks
            .iter()
            .any(|n| n.id == network)
        {
            return Err(invalid("unknown routing subproblem"));
        }
        self.restricted(Restriction::RoutingMultiplicity(network, bounds))
    }
    /// Shared additive restriction in original coordinates and physical arcs.
    pub fn with_row(
        &self,
        row: Row,
        routing: &[RowCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.validate_original(original)?;
        let mut seen = HashSet::new();
        for c in routing {
            let network = self
                .definition
                .problem
                .networks
                .iter()
                .find(|n| n.id == c.network)
                .ok_or_else(|| invalid("unknown shared-row routing subproblem"))?;
            if !seen.insert(c.network)
                || c.arcs.len() != network.graph.arcs.len()
                || !c.per_route.is_finite()
                || c.arcs.iter().any(|a| !a.is_finite())
            {
                return Err(invalid("shared-row routing coefficients"));
            }
        }
        self.restricted(Restriction::SharedRow {
            row,
            routing: routing.to_vec(),
            original: original.to_vec(),
        })
    }
    /// Shared nonlinear routing presence coupled to original coordinates.
    pub fn with_presence_row(
        &self,
        row: Row,
        routing: &[PresenceCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Presence(routing.to_vec()),
            original,
        )
    }
    pub(in crate::master::multi_pricing::imported) fn with_nonlinear_row(
        &self,
        row: Row,
        kernels: nonlinear::RowKernels,
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.validate_original(original)?;
        kernels.validate(&self.definition.problem, &row)?;
        self.restricted(Restriction::NonlinearRow {
            row,
            kernels,
            original: original.to_vec(),
        })
    }
    fn validate_original(&self, original: &[(i32, f64)]) -> Result<()> {
        let mut seen = HashSet::new();
        if original.iter().any(|(id, a)| {
            !seen.insert(*id)
                || !a.is_finite()
                || !self.definition.variables.iter().any(|v| v.id == *id)
        }) {
            return Err(invalid("coalesced original-variable coefficients required"));
        }
        Ok(())
    }
}
