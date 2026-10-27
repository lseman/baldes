//! Routing kernels coupled to original coordinates across import rebuilds.
use super::*;

impl<const D: usize> ImportedSnapshot<D> {
    /// Retain deterministic coupled activity rows across imported-domain rebuilds.
    pub fn with_accumulated_activity_row(
        &self,
        row: Row,
        routing: &[AccumulatedActivityCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Accumulated(routing.to_vec()),
            original,
        )
    }

    /// Install registered routing coefficients coupled to original coordinates.
    /// The descriptors and row survive later imported-domain extensions.
    pub fn with_registered_cut_row(
        &self,
        row: Row,
        routing: &[RegisteredCutCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Registered {
                coefficients: routing.to_vec(),
                composite: false,
            },
            original,
        )
    }

    /// Sum registered kernels, including repeated descriptors on one network.
    pub fn with_composite_cut_row(
        &self,
        row: Row,
        routing: &[RegisteredCutCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Registered {
                coefficients: routing.to_vec(),
                composite: true,
            },
            original,
        )
    }

    /// Couple monotone binary-grid resource-load steps to original coordinates.
    /// The routing adapter's load, window and row-sense requirements apply.
    pub fn with_resource_knapsack_row(
        &self,
        row: Row,
        routing: &[ResourceKnapsackCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Resource(routing.to_vec()),
            original,
        )
    }

    fn with_nonlinear_row(
        &self,
        row: Row,
        kernels: nonlinear::RowKernels,
        original: &[(i32, f64)],
    ) -> Result<Self> {
        if original
            .iter()
            .any(|(id, _)| !self.context.registry.original.contains(id))
        {
            return Err(invalid("unknown imported original variable"));
        }
        let node = match &kernels {
            nonlinear::RowKernels::Registered {
                coefficients,
                composite: true,
            } => self
                .node
                .with_composite_cut_master_row(row.clone(), coefficients, original)?,
            nonlinear::RowKernels::Registered {
                coefficients,
                composite: false,
            } => self
                .node
                .with_registered_cut_master_row(row.clone(), coefficients, original)?,
            nonlinear::RowKernels::Accumulated(c) => self
                .node
                .with_accumulated_activity_master_row(row.clone(), c, &[], original)?,
            nonlinear::RowKernels::Resource(c) => {
                self.node
                    .with_resource_knapsack_master_row(row.clone(), c, original)?
            }
            nonlinear::RowKernels::Presence(c) => {
                self.node
                    .with_presence_master_row(row.clone(), c, original)?
            }
        };
        Ok(Self {
            node,
            context: self.context.restricted(Restriction::NonlinearRow {
                row,
                kernels,
                original: original.to_vec(),
            }),
        })
    }
}

impl<const D: usize> MultiImport<D> {
    /// Retain deterministic coupled activity rows across imported-domain rebuilds.
    pub fn with_accumulated_activity_row(
        &self,
        row: Row,
        routing: &[AccumulatedActivityCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Accumulated(routing.to_vec()),
            original,
        )
    }

    /// Retain registered routing kernels through pricing and tree descendants.
    pub fn with_registered_cut_row(
        &self,
        row: Row,
        routing: &[RegisteredCutCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Registered {
                coefficients: routing.to_vec(),
                composite: false,
            },
            original,
        )
    }

    /// Sum routing kernels; repeated descriptors contribute repeatedly.
    pub fn with_composite_cut_row(
        &self,
        row: Row,
        routing: &[RegisteredCutCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Registered {
                coefficients: routing.to_vec(),
                composite: true,
            },
            original,
        )
    }

    /// Retain resource-load steps through phase I and later source batches.
    pub fn with_resource_knapsack_row(
        &self,
        row: Row,
        routing: &[ResourceKnapsackCoefficients],
        original: &[(i32, f64)],
    ) -> Result<Self> {
        self.with_nonlinear_row(
            row,
            nonlinear::RowKernels::Resource(routing.to_vec()),
            original,
        )
    }
}
