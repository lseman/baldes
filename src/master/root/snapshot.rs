use super::*;

impl<const D: usize> std::fmt::Debug for NodeSnapshot<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSnapshot")
            .field("rows", &self.master.rhs.len())
            .field("columns", &self.master.costs.len())
            .field("cuts", &self.active_cuts())
            .finish()
    }
}

impl<const D: usize> NodeSnapshot<D> {
    pub fn bound_mode(&self) -> BoundMode {
        self.warm.bound_mode
    }
    pub fn master_solver(&self) -> MasterSolver {
        self.master.solver()
    }
    pub fn branch_depth(&self) -> usize {
        self.problem.branch_constraints.len()
    }
    pub fn active_cuts(&self) -> usize {
        self.cut.rows.iter().filter(|r| r.is_some()).count()
            + self
                .cut
                .capacity_rows
                .iter()
                .filter(|r| r.is_some())
                .count()
            + self
                .cut
                .knapsack_rows
                .iter()
                .filter(|r| r.is_some())
                .count()
    }
    pub fn enumerated_routes(&self) -> Option<usize> {
        self.pricing.enumeration.as_ref().map(|e| e.paths.len)
    }
    pub(super) fn fork(
        &self,
        branches: &[BranchConstraint],
        penalty: f64,
    ) -> Result<(RootProblem<D>, ClpMaster, cg::WarmStart)> {
        let mut master = self.master.try_clone()?;
        let mut problem = self.problem.clone();
        let mut warm = self.warm.clone();
        let penalty = warm
            .artificials
            .iter()
            .map(|&c| master.costs[c])
            .fold(penalty, f64::max);
        for constraint in branches {
            constraint.validate(&problem)?;
            if problem.tighten_vehicle_upper(constraint)? {
                let mut rhs = master.rhs.clone();
                rhs[problem.row_map.borrow()[0]] = problem.rows[0].rhs;
                master.set_rhs(&rhs)?;
                continue;
            }
            let (sense, mut rhs) = constraint.row();
            rhs -= problem.fixed_sum(|route| Ok(constraint.coefficient(&problem, route)))?;
            let mut entries = vec![];
            for (c, column) in warm.columns.iter_mut().enumerate() {
                let Some(column) = column else {
                    continue;
                };
                let route = &self.routes[column.tag as usize];
                if crate::master::branching::enforcement::retires_column(constraint, route) {
                    master.set_upper(c, 0.)?;
                }
                let coefficient = constraint.coefficient(&problem, route);
                if coefficient != 0. {
                    entries.push((c, coefficient));
                    column.entries.push((master.rhs.len(), coefficient));
                }
            }
            for column in &mut warm.generated_pool {
                let route = &self.routes[column.tag as usize];
                let coefficient = constraint.coefficient(&problem, route);
                if coefficient != 0. {
                    column.entries.push((master.rhs.len(), coefficient));
                }
            }
            let row = master.add_row(sense, rhs, &entries)?;
            problem.append_constraint(constraint.clone(), row)?;
            if penalty > 0. {
                for sign in [1., -1.] {
                    if (sign > 0. && sense == RowSense::LessEqual)
                        || (sign < 0. && sense == RowSense::GreaterEqual)
                    {
                        continue;
                    }
                    let c = master.add_column(penalty, &[(row, sign)])?;
                    warm.columns.push(None);
                    warm.artificials.push(c);
                }
            }
        }
        if !branches.is_empty() {
            warm.mark_proximal_child();
        }
        Ok((problem, master, warm))
    }
    pub(crate) fn restricted_value(
        &self,
        branches: &[BranchConstraint],
        penalty: f64,
    ) -> Result<(f64, usize)> {
        let (_, mut master, _) = self.fork(branches, penalty)?;
        let value = if master.solve() {
            master.objective
        } else {
            1e12
        };
        Ok((value, master.simplex_iterations))
    }
}
