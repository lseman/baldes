//! Zero-objective restricted phase I for external column generation.
use super::*;

impl<const D: usize> ColumnImport<D> {
    /// Minimize artificial activity with all real objective coefficients zero.
    /// Positive restricted violation is not external-domain infeasibility.
    pub fn feasibility_pricing(&self, options: Options) -> Result<ImportPricing<D>> {
        self.feasibility_point(options, false)
    }
    pub(in crate::master::model::imported) fn feasibility_point(
        &self,
        mut options: Options,
        certify: bool,
    ) -> Result<ImportPricing<D>> {
        options.preprocessing.enabled = false;
        options.enumeration = None;
        options.cg.stabilize = false;
        options.cg.bound_mode = cg::BoundMode::Numerical;
        options.cg.artificial_cost = 1.;
        options.cg.max_penalty_updates = 0;
        let mut compiled = lifecycle::compile_definition(&self.definition, Some(&self.context))?;
        let registry = Rc::new(compiled.registry);
        for variable in &mut compiled.variables {
            variable.cost = 0.;
        }
        for network in &mut compiled.problem.networks {
            network.graph.suppress_coupled_costs();
            for arc in &mut network.graph.arcs {
                arc.cost = 0.;
            }
        }
        for fixed in &mut compiled.fixed {
            fixed.route.route.original_cost = 0.;
        }
        let output = solve_with_fixed_routes(
            compiled.problem,
            compiled.variables,
            options,
            &compiled.fixed,
            &compiled.nonlinear,
        )?;
        let master = &output.snapshot.master;
        if !master.solved
            || master.status != 0
            || matches!(
                output.master.status,
                cg::Status::TimeLimit | cg::Status::IterationLimit
            )
            || output.duals.iter().any(|x| !x.is_finite())
        {
            return Err(crate::incomplete(
                "optimal restricted phase-I master unavailable",
            ));
        }
        let mut violation = 0.;
        for &column in &output.master.artificials {
            let value = master.values[column];
            if !value.is_finite() || value < -1e-7 {
                return Err(crate::incomplete(
                    "invalid restricted phase-I artificial value",
                ));
            }
            violation += value.max(0.);
        }
        if !violation.is_finite() {
            return Err(crate::incomplete("restricted phase-I violation overflow"));
        }
        let retained_bound = if certify {
            super::certificate::retained_bound(self, &output)
        } else {
            None
        };
        Ok(ImportPricing {
            epoch: Rc::new(()),
            context: self.context.clone(),
            definition: self.definition.clone(),
            registry,
            position: self.position,
            duals: Rc::new(output.duals),
            shared_offset: output.snapshot.state.shared_offset,
            kind: ImportPricingKind::Feasibility,
            violation: Some(violation),
            routing: Rc::new(output.routes),
            retained_bound,
            open_groups: Rc::new(self.open_groups.clone()),
        })
    }
}
