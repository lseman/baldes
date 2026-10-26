//! Objective-price recovery without finite artificial-penalty assumptions.
use super::*;
use crate::master::clp::ClpSolveOutcome;

fn remove_artificials<const D: usize>(output: &mut Output<D>, deadline: Instant) -> Result<bool> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Ok(false);
    }
    let master = &mut output.snapshot.master;
    for &column in &output.master.artificials {
        master.set_upper(column, 0.)?;
    }
    for &(_, column) in &output.master.warm_start.cut_artificials {
        master.set_upper(column, 0.)?;
    }
    if master.solve_limited(i32::MAX as usize, remaining)? != ClpSolveOutcome::Optimal {
        return Ok(false);
    }
    output.duals = master.duals.clone();
    // The zero-artificial re-solve can change the primal basis. Keep exported
    // coordinates and physical route quantities at that same point.
    output.values.fill(0.);
    output
        .master_values
        .iter_mut()
        .for_each(|(_, value)| *value = 0.);
    for (metadata, &value) in output.snapshot.warm.columns.iter().zip(&master.values) {
        let Some(metadata) = metadata else { continue };
        if let Some((index, sign)) = super::super::super::master_variables::decode(metadata.tag) {
            output.master_values[index].1 += sign * value;
        } else if metadata.tag >= 0 {
            output.values[metadata.tag as usize] += value;
        }
    }
    Ok(true)
}
impl<const D: usize> ColumnImport<D> {
    /// Obtain an optimal numerical point for the current restricted master,
    /// fixing artificial quantities to zero. Missing feasible routing paths
    /// may be restored from phase I. No external-domain proof is published.
    pub fn pricing(&self, options: Options) -> Result<ImportPricing<D>> {
        self.objective_point(options, false)
    }
    pub(super) fn objective_point(
        &self,
        options: Options,
        certify: bool,
    ) -> Result<ImportPricing<D>> {
        self.objective_point_seeded(options, certify, &[])
    }

    pub(super) fn objective_point_seeded(
        &self,
        options: Options,
        certify: bool,
        seeds: &[NetworkRoute],
    ) -> Result<ImportPricing<D>> {
        let (output, registry) = self.objective_output_seeded(options, seeds)?;
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
            kind: ImportPricingKind::Objective,
            violation: None,
            routing: Rc::new(output.routes),
            retained_bound,
            open_groups: Rc::new(self.open_groups.clone()),
        })
    }
    /// Original-cost feasible RMP, also used by the external-domain tree.
    pub(in crate::master::model::imported) fn objective_output(
        &self,
        options: Options,
    ) -> Result<(Output<D>, Rc<Registry>)> {
        self.objective_output_seeded(options, &[])
    }

    fn objective_output_seeded(
        &self,
        mut options: Options,
        seeds: &[NetworkRoute],
    ) -> Result<(Output<D>, Rc<Registry>)> {
        options.preprocessing.enabled = false;
        options.enumeration = None;
        options.cg.stabilize = false;
        options.cg.bound_mode = cg::BoundMode::Numerical;
        let deadline = Instant::now()
            .checked_add(options.cg.time_limit)
            .ok_or_else(|| invalid("import pricing time limit"))?;
        let compiled = lifecycle::compile_definition(&self.definition, Some(&self.context))?;
        let mut registry = Rc::new(compiled.registry);
        let mut initial_options = options.clone();
        initial_options.cg.time_limit = deadline.saturating_duration_since(Instant::now());
        let mut output = solve_with_seed_routes(
            compiled.problem,
            compiled.variables,
            initial_options,
            &compiled.fixed,
            &compiled.nonlinear,
            seeds,
        )?;
        if matches!(
            output.master.status,
            cg::Status::TimeLimit | cg::Status::IterationLimit | cg::Status::Unbounded
        ) {
            return Err(crate::incomplete(
                "optimal feasible restricted import master unavailable",
            ));
        }
        if !remove_artificials(&mut output, deadline)? {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(crate::incomplete("import pricing time limit"));
            }
            let mut phase_options = options.clone();
            phase_options.cg.time_limit = remaining;
            let phase = self.feasibility_pricing(phase_options)?;
            if phase.restricted_violation() != Some(0.) {
                return Err(crate::incomplete(
                    "feasible restricted import master unavailable",
                ));
            }
            let compiled = lifecycle::compile_definition(&self.definition, Some(&self.context))?;
            registry = Rc::new(compiled.registry);
            options.cg.time_limit = deadline.saturating_duration_since(Instant::now());
            output = solve_with_seed_routes(
                compiled.problem,
                compiled.variables,
                options,
                &compiled.fixed,
                &compiled.nonlinear,
                &phase.routing,
            )?;
            if matches!(
                output.master.status,
                cg::Status::TimeLimit | cg::Status::IterationLimit
            ) || !remove_artificials(&mut output, deadline)?
            {
                return Err(crate::incomplete(
                    "optimal feasible restricted import master unavailable",
                ));
            }
        }
        if output.duals.iter().any(|x| !x.is_finite()) {
            return Err(crate::incomplete("nonfinite restricted import duals"));
        }
        Ok((output, registry))
    }
}
