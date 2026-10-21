use super::*;

/// Cumulative work performed by a persistent solver, including cold retries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SolverStatistics {
    pub symbolic_analyses: usize,
    pub symbolic_extensions: usize,
    pub warm_starts: usize,
    pub cold_restarts: usize,
    pub newton_iterations: usize,
}

/// Reusable IPM preparation and strictly interior starting points.
/// Matrix/bound/sense or scaling changes rebuild preparation. Cost and RHS
/// edits retain symbolic analysis but start cold. Structural changes always
/// discard the previous iterate; compatible symbolic structure may be reused.
#[derive(Debug, Clone, Default)]
pub struct Solver {
    prepared: Option<Prepared>,
    statistics: SolverStatistics,
}

#[derive(Debug, Clone)]
pub(super) struct Prepared {
    model: ModelData,
    scaling_iterations: usize,
    kind: SolverType,
    pub(super) f: StandardForm,
    pub(super) row_scale: Vec<f64>,
    pub(super) col_scale: Vec<f64>,
    pub(super) cost_scale: f64,
    pub(super) unscaled_cost_norm: f64,
    pub(super) ls: SparseSolver,
    pub(super) solve_workspace: super::super::sparse_solver::SolveWorkspace,
    pub(super) iterations: usize,
    last: Option<Point>,
}

impl Prepared {
    fn compatible(&self, model: &ModelData, options: &Options) -> bool {
        self.model.a == model.a
            && self.model.lower == model.lower
            && self.model.upper == model.upper
            && self.model.sense == model.sense
            && self.scaling_iterations == options.scaling_iterations
            && self.kind == options.solver
    }

    fn new(model: &ModelData, options: &Options, previous: Option<&Self>) -> Result<(Self, bool)> {
        let mut f = model.standard_form()?;
        if let Some(previous) = previous {
            Self::retain_column_order(&mut f, model, previous);
        }
        let (row_scale, col_scale) = scale(&mut f, options.scaling_iterations)?;
        let extended = previous
            .filter(|p| p.kind == options.solver)
            .map(|p| p.ls.with_appended_columns(f.a.clone(), options.solver))
            .transpose()?
            .flatten();
        let reused = extended.is_some();
        let ls = match extended {
            Some(ls) => ls,
            None => SparseSolver::new(f.a.clone(), options.solver)?,
        };
        let mut out = Self {
            model: model.clone(),
            scaling_iterations: options.scaling_iterations,
            kind: options.solver,
            f,
            row_scale,
            col_scale,
            ls,
            solve_workspace: Default::default(),
            cost_scale: 1.,
            unscaled_cost_norm: 0.,
            iterations: 0,
            last: None,
        };
        out.update_vectors(model)?;
        Ok((out, reused))
    }

    fn retain_column_order(f: &mut StandardForm, model: &ModelData, previous: &Self) {
        let old = &previous.model;
        let old_columns = old.a.cols;
        if model.a.cols <= old_columns
            || model.sense != old.sense
            || model.lower[..old_columns] != old.lower
            || model.upper[..old_columns] != old.upper
            || model.a.ptr[..=old_columns] != old.a.ptr
            || model.a.idx[..old.a.nnz()] != old.a.idx
        {
            return;
        }
        // Standard form normally places inequality slacks last. Keep their
        // existing indices and append new transformed variables after them,
        // otherwise every CG insertion would invalidate the symbolic prefix.
        let mut permutation = vec![usize::MAX; previous.f.c.len()];
        let mut used = vec![false; f.c.len()];
        let mut new_slack_by_row = vec![usize::MAX; f.b.len()];
        let original_terms: usize = f.recovery.iter().map(|(_, terms)| terms.len()).sum();
        for j in original_terms..f.c.len() {
            let (row, _) = f.a.column(j).next().unwrap();
            new_slack_by_row[row] = j;
        }
        for (old, new) in previous.f.recovery.iter().zip(&f.recovery) {
            for (&(before, _), &(after, _)) in old.1.iter().zip(&new.1) {
                permutation[before] = after;
                used[after] = true;
            }
        }
        for (j, mapped) in permutation.iter_mut().enumerate() {
            if *mapped == usize::MAX {
                let (row, _) = previous.f.a.column(j).next().unwrap();
                *mapped = new_slack_by_row[row];
                used[*mapped] = true;
            }
        }
        permutation.extend((0..f.c.len()).filter(|&j| !used[j]));
        let mut inverse = vec![0; permutation.len()];
        let mut ptr = Vec::with_capacity(permutation.len() + 1);
        let mut idx = Vec::with_capacity(f.a.nnz());
        let mut val = Vec::with_capacity(f.a.nnz());
        let mut cost = Vec::with_capacity(permutation.len());
        ptr.push(0);
        for (j, &source) in permutation.iter().enumerate() {
            inverse[source] = j;
            cost.push(f.c[source]);
            idx.extend_from_slice(&f.a.idx[f.a.ptr[source]..f.a.ptr[source + 1]]);
            val.extend_from_slice(&f.a.val[f.a.ptr[source]..f.a.ptr[source + 1]]);
            ptr.push(idx.len());
        }
        f.a.ptr = ptr;
        f.a.idx = idx;
        f.a.val = val;
        f.c = cost;
        for (j, _) in &mut f.upper {
            *j = inverse[*j];
        }
        for (_, terms) in &mut f.recovery {
            for (j, _) in terms {
                *j = inverse[*j];
            }
        }
    }

    fn update_vectors(&mut self, model: &ModelData) -> Result<()> {
        self.f.b.clone_from(&model.b);
        for (b, sign) in self.f.b.iter_mut().zip(&self.f.row_sign) {
            *b *= sign;
        }
        self.f.c.fill(0.);
        for (j, (offset, terms)) in self.f.recovery.iter().enumerate() {
            if *offset != 0. {
                for (i, a) in model.a.column(j) {
                    self.f.b[i] -= self.f.row_sign[i] * a * offset;
                }
            }
            for &(k, sign) in terms {
                self.f.c[k] = model.c[j] * sign * self.col_scale[k];
            }
        }
        for (b, scale) in self.f.b.iter_mut().zip(&self.row_scale) {
            *b *= scale;
        }
        if !finite(&self.f.b) || !finite(&self.f.c) {
            return Err(Error::NumericalFailure(
                "equilibration overflow or underflow".into(),
            ));
        }
        self.unscaled_cost_norm = self
            .f
            .c
            .iter()
            .zip(&self.col_scale)
            .fold(0.0_f64, |v, (c, s)| v.max((c / s).abs()));
        self.cost_scale = 1. / norm(&self.f.c).max(1.);
        for c in &mut self.f.c {
            *c *= self.cost_scale;
        }
        self.model.b.clone_from(&model.b);
        self.model.c.clone_from(&model.c);
        Ok(())
    }
}

impl Solver {
    pub fn statistics(&self) -> SolverStatistics {
        self.statistics
    }

    pub fn solve(&mut self, model: &ModelData, options: &Options) -> Result<Solution> {
        if options
            .gap_tolerance
            .is_some_and(|gap| !gap.is_finite() || gap <= 0. || gap > 1.)
        {
            return Err(Error::InvalidInput(
                "gap tolerance must be finite and in (0, 1]".into(),
            ));
        }
        if !options.tolerance.is_finite() || options.tolerance <= 0. || options.tolerance >= 1. {
            return Err(Error::InvalidInput(
                "tolerance must be finite and between zero and one".into(),
            ));
        }
        model.validate()?;
        let started = std::time::Instant::now();
        let reusable = self
            .prepared
            .as_ref()
            .is_some_and(|p| p.compatible(model, options));
        let mut initial = None;
        if reusable {
            let p = self.prepared.as_mut().unwrap();
            if p.model.b == model.b && p.model.c == model.c {
                initial = p.last.as_ref().cloned();
            }
            p.last = None;
            p.update_vectors(model)?;
        } else {
            // Discard stale preparation even if building the replacement fails.
            let previous = self.prepared.take();
            let (mut p, extended) = Prepared::new(model, options, previous.as_ref())?;
            if let Some(previous) = previous {
                p.solve_workspace = previous.solve_workspace;
            }
            self.prepared = Some(p);
            if extended {
                self.statistics.symbolic_extensions += 1;
            } else {
                self.statistics.symbolic_analyses += 1;
            }
        }
        let warm = initial.is_some();
        self.statistics.warm_starts += usize::from(warm);
        let p = self.prepared.as_mut().unwrap();
        p.iterations = 0;
        let mut budget = options.clone();
        budget.time_limit = options
            .time_limit
            .map(|t| t.saturating_sub(started.elapsed()));
        if warm {
            budget.max_iterations = budget.max_iterations.min(20);
        }
        let mut result = solve_prepared(model, &budget, p, initial);
        let first_iterations = p.iterations;
        self.statistics.newton_iterations += first_iterations;
        let retry = warm
            && match &result {
                Ok((s, _)) => s.status != Status::Optimal && s.status != Status::TimeLimit,
                Err(Error::NumericalFailure(_)) => true,
                _ => false,
            };
        if retry
            && first_iterations < options.max_iterations
            && options.time_limit.map_or(true, |t| started.elapsed() < t)
        {
            self.statistics.cold_restarts += 1;
            budget.max_iterations = options.max_iterations - first_iterations;
            budget.time_limit = options
                .time_limit
                .map(|t| t.saturating_sub(started.elapsed()));
            p.iterations = 0;
            result = solve_prepared(model, &budget, p, None);
            self.statistics.newton_iterations += p.iterations;
            if let Ok((s, _)) = &mut result {
                s.iterations += first_iterations;
            }
        }
        let (solution, point) = result?;
        if solution.status == Status::Optimal {
            p.last = Some(point);
        }
        Ok(solution)
    }
}
