use super::*;

#[derive(Debug)]
pub struct ImportedSnapshot<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) node: NodeSnapshot<D>,
    pub(super) context: Rc<Context<D>>,
}
impl<const D: usize> ImportedSnapshot<D> {
    /// Ordinary snapshot operations, including preprocessing, tree and MIP.
    pub fn node(&self) -> &NodeSnapshot<D> {
        &self.node
    }
    /// Decode original and imported-column values returned by this lineage's
    /// ordinary snapshot tree/MIP operations. Internal IDs are not column IDs.
    pub fn values(&self, master_values: &[(i32, f64)]) -> Result<Values> {
        self.context.registry.values(master_values)
    }
    pub fn solve(&self, options: Options) -> Result<ImportedOutput<D>> {
        ImportedOutput::new(self.node.solve(options)?, self.context.clone())
    }
    /// Restrict total route multiplicity in a routing subproblem. Unlike a
    /// restriction on a separate low-level child, this survives import extension.
    pub fn with_routing_multiplicity(&self, network: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        Ok(Self {
            node: self.node.with_multiplicity(network, bounds)?,
            context: self
                .context
                .restricted(Restriction::RoutingMultiplicity(network, bounds)),
        })
    }
    /// Fix additional copies of a registered routing path, preserving the
    /// physical path and quantity across later imported-domain extensions.
    pub fn fix_route(&self, route: usize, quantity: f64) -> Result<Self> {
        let node = self.node.fix_route(route, quantity)?;
        let fixed = node.state.fixed_routes.last().unwrap().clone();
        Ok(Self {
            node,
            context: self.context.restricted(Restriction::FixedRoute(fixed)),
        })
    }
    /// Add a shared restriction using original coordinate IDs and physical
    /// routing coefficients. The row also applies to subsequently added columns
    /// through their declared original vectors. Internal column IDs are rejected.
    pub fn with_row(
        &self,
        row: Row,
        routing: &[RowCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        if original
            .iter()
            .any(|(id, _)| !self.context.registry.original.contains(id))
        {
            return Err(invalid("unknown imported original variable"));
        }
        let node = self.node.with_master_row(row.clone(), routing, original)?;
        Ok(Self {
            node,
            context: self.context.restricted(Restriction::SharedRow {
                row,
                routing: routing.to_vec(),
                original: original.to_vec(),
            }),
        })
    }
    /// Couple original coordinates to nonlinear routing-event presence. The
    /// original descriptor is retained across import extension and fixed replay.
    pub fn with_presence_row(
        &self,
        row: Row,
        routing: &[PresenceCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        if original
            .iter()
            .any(|(id, _)| !self.context.registry.original.contains(id))
        {
            return Err(invalid("unknown imported original variable"));
        }
        let node = self
            .node
            .with_presence_master_row(row.clone(), routing, original)?;
        Ok(Self {
            node,
            context: self.context.restricted(Restriction::NonlinearRow {
                row,
                kernels: nonlinear::RowKernels::Presence(routing.to_vec()),
                original: original.to_vec(),
            }),
        })
    }
    pub fn with_variable_bounds(&self, id: i32, bounds: Bounds) -> Result<Self> {
        if !self.context.registry.original.contains(&id) {
            return Err(invalid("unknown imported original variable"));
        }
        Ok(Self {
            node: self.node.with_master_variable_bounds(id, bounds)?,
            context: self.context.restricted(Restriction::Variable(id, bounds)),
        })
    }
    pub fn with_column_bounds(&self, subproblem: i32, column: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        let id = *self
            .context
            .registry
            .columns
            .get(&(subproblem, column))
            .ok_or_else(|| invalid("unknown imported column"))?;
        Ok(Self {
            node: self.node.with_master_variable_bounds(id, bounds)?,
            context: self
                .context
                .restricted(Restriction::Column(subproblem, column, bounds)),
        })
    }
    /// Total multiplicity includes fixed copies because every imported column
    /// remains an original master variable with its full contribution.
    pub fn with_multiplicity(&self, subproblem: i32, bounds: Bounds) -> Result<Self> {
        quantity_bounds(bounds)?;
        let ids = self
            .context
            .registry
            .groups
            .get(&subproblem)
            .ok_or_else(|| invalid("unknown imported subproblem"))?;
        let coefficients: Vec<_> = ids.iter().map(|&id| (id, 1.)).collect();
        let mut node = self.node.fork()?;
        for (sense, rhs) in [
            (RowSense::GreaterEqual, bounds.lower),
            (RowSense::LessEqual, bounds.upper),
        ] {
            if rhs.is_finite() {
                node = node.with_master_row(Row { sense, rhs }, &[], &coefficients)?;
            }
        }
        Ok(Self {
            node,
            context: self
                .context
                .restricted(Restriction::Multiplicity(subproblem, bounds)),
        })
    }
}
