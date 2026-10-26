//! Original unit-cover bounds from a complete conservative path relaxation.
//! The auxiliary LP supplies duals only. Its objective/status is never a proof.
use super::*;
use crate::enumeration::PathDomainLimits;
use crate::master::certificates::{Column as ProofColumn, FiniteMaster};
use crate::master::clp::ClpSolveOutcome;
use std::time::{Duration, Instant};

/// Scoped to one immutable original problem, never inherited by child nodes.
/// A deterministic domain failure cannot be repaired by new duals or cuts.
/// Keep the work budget so an increased budget can still retry the search.
#[derive(Default)]
pub(super) struct SearchCache {
    failed_domain_limits: std::cell::Cell<Option<(usize, usize, usize)>>,
    #[cfg(test)]
    attempts: std::cell::Cell<usize>,
}
impl SearchCache {
    fn collect<'a, const D: usize>(
        &self,
        problem: &'a RoutingContext<D>,
        limits: PathDomainLimits,
    ) -> Option<ConservativeCoverRelaxation<'a, D>> {
        use super::domain::CoverDomainFailure;
        use crate::enumeration::PathDomainFailure;
        if self
            .failed_domain_limits
            .get()
            .is_some_and(|(work, paths, arcs)| {
                limits.max_work <= work
                    && limits.max_paths <= paths
                    && limits.max_stored_arcs <= arcs
            })
        {
            return None;
        }
        #[cfg(test)]
        self.attempts.set(self.attempts.get() + 1);
        match ConservativeCoverRelaxation::collect(problem, limits) {
            Ok(domain) => Some(domain),
            Err(failure) => {
                // A transient deadline/allocation failure is deliberately retried.
                if !matches!(
                    failure,
                    CoverDomainFailure::Paths(
                        PathDomainFailure::TimeLimit | PathDomainFailure::Allocation
                    )
                ) {
                    self.failed_domain_limits.set(Some((
                        limits.max_work,
                        limits.max_paths,
                        limits.max_stored_arcs,
                    )));
                }
                None
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/routing/certificate/search_tests.rs"]
mod search_tests;

struct Budget {
    deadline: Instant,
    work: usize,
    entries: usize,
}
impl Budget {
    fn spend(&mut self, work: usize) -> Option<()> {
        if Instant::now() >= self.deadline {
            return None;
        }
        self.work = self.work.checked_sub(work)?;
        Some(())
    }
    fn store(&mut self, entries: usize) -> Option<()> {
        self.spend(entries)?;
        self.entries = self.entries.checked_sub(entries)?;
        Some(())
    }
}

/// Unknown on unsupported geometry, exhausted budget, or incomplete proof.
/// No parent model, pricing cache, or primal witness is changed here.
pub(super) fn lower_bound<const D: usize>(
    problem: &RoutingContext<D>,
    state: &State<D>,
    master: &ClpMaster,
    cache: &SearchCache,
) -> Option<f64> {
    if problem.model.networks[0].graph.has_coupled_resources() {
        return None;
    }
    let now = Instant::now();
    let deadline = now
        .checked_add(Duration::from_millis(250))?
        .min(state.deadline.unwrap_or(now + Duration::from_secs(1)));
    let mut budget = Budget {
        deadline,
        work: 10_000_000,
        entries: 1_000_000,
    };
    budget.spend(master.rhs.len())?;
    let map = problem.row_map.borrow();
    if map.len() != problem.model.rows.len() {
        return None;
    }
    let mut logical = vec![false; master.rhs.len()];
    for (row, &physical) in problem.model.rows.iter().zip(map.iter()) {
        budget.spend(1)?;
        if physical >= logical.len()
            || logical[physical]
            || row.rhs != master.rhs[physical]
            || row.sense != master.senses[physical]
        {
            return None;
        }
        logical[physical] = true;
    }
    // Residual transformations of arbitrary fractional rows need a separate
    // exact shift proof. Integer branch/unit-cover rows have exact shifts.
    if !problem.fixed_routes.is_empty() {
        for row in &problem.model.rows {
            if row.rhs.fract() != 0. || row.rhs.abs() > (1u64 << 30) as f64 {
                return None;
            }
        }
        for &(row, value) in problem.model.networks[0]
            .route_entries
            .iter()
            .chain(problem.model.networks[0].arc_entries.iter().flatten())
        {
            budget.spend(1)?;
            if row >= map.len() || value.fract() != 0. || value.abs() > 1. {
                return None;
            }
        }
    }
    let domain = cache.collect(
        problem,
        PathDomainLimits {
            max_paths: 10_000,
            max_stored_arcs: budget.entries,
            max_work: budget.work,
            deadline: Some(deadline),
        },
    )?;
    budget.spend(domain.work_used())?;
    let mut costs = Vec::new();
    let mut entries = Vec::new();
    let mut direct = Vec::new();
    // Sparse transpose of optional rows: extracting each row must not scan
    // every other row's coefficients again.
    let mut optional_rows = vec![Vec::<(usize, i64)>::new(); logical.len()];
    for path in domain.paths().paths() {
        budget.store(path.arc_positions().len())?;
        let mut raw_costs = Vec::new();
        let mut raw_entries = Vec::new();
        let mut nodes = vec![i32::try_from(problem.model.networks[0].graph.source).ok()?];
        let mut arcs = Vec::new();
        for &position in path.arc_positions() {
            budget.spend(1)?;
            let arc = &problem.model.networks[0].graph.arcs[position];
            raw_costs.push(arc.cost);
            arcs.push(arc.id);
            nodes.push(i32::try_from(arc.head).ok()?);
            for &(row, value) in &problem.model.networks[0].arc_entries[position] {
                budget.store(1)?;
                raw_entries.push((*map.get(row)?, value));
            }
        }
        for &(row, value) in &problem.model.networks[0].route_entries {
            budget.store(1)?;
            raw_entries.push((*map.get(row)?, value));
        }
        // Lower resource coordinates describe an auxiliary column, not an
        // actual feasible route. Never replay/filter them with floating tests.
        let route = Route {
            nodes,
            arcs,
            cost: 0.,
            original_cost: 0.,
            resources: path.resource_lower_bounds().last()?
                [..problem.model.networks[0].graph.resources.len()]
                .to_vec(),
            resource_profile: vec![],
        };
        for (row, constraint) in &problem.branch_constraints {
            budget.spend(1)?;
            if matches!(constraint, BranchConstraint::RyanFoster(_)) {
                budget.store(1)?;
                raw_entries.push((map[*row], constraint.coefficient(problem, &route)));
            }
        }
        let cut_work = state
            .solver
            .cuts
            .cuts
            .len()
            .checked_add(state.cut.capacity_cuts.len())?
            .checked_add(state.solver.cuts.knapsack_cuts.len())?
            .checked_mul(route.nodes.len())?;
        budget.spend(cut_work)?;
        let cut_entries = state.cut.entries(&state.solver, &route, true).ok()?;
        budget.store(cut_entries.len().checked_mul(2)?)?;
        for &(row, value) in &cut_entries {
            if row >= logical.len()
                || logical[row]
                || !value.is_finite()
                || value.fract() != 0.
                || value.abs() > (1u64 << 53) as f64
            {
                return None;
            }
            optional_rows[row].push((entries.len(), value as i64));
        }
        raw_entries.extend(cut_entries);
        direct.push(route.nodes.len() == 2);
        costs.push(raw_costs);
        entries.push(raw_entries);
    }
    // A cut is usable only when it holds for every residual integer partition
    // of the auxiliary paths. Thus schedule-dependent coefficients need no
    // assumed equality with the native schedule: the whole auxiliary cover
    // is itself a feasible image. Empty routes must contribute zero.
    let mut keep = logical.clone();
    for row in 0..keep.len() {
        budget.spend(1)?;
        if logical[row] {
            continue;
        }
        budget.spend(entries.len())?;
        let mut coefficients = vec![0i64; entries.len()];
        for &(path, value) in &optional_rows[row] {
            budget.spend(1)?;
            coefficients[path] = coefficients[path].checked_add(value)?;
        }
        if direct
            .iter()
            .zip(&coefficients)
            .any(|(&empty, &c)| empty && c != 0)
        {
            continue;
        }
        // An additive envelope proves packing rank-one rows in linear path
        // work. Verify actual auxiliary coefficients rather than trusting the
        // separator, memory mode, or cut support to supply a valid row.
        if master.senses[row] == RowSense::LessEqual {
            budget.spend(state.cut.rows.len())?;
            if let Some(index) = state.cut.rows.iter().position(|&r| r == Some(row)) {
                let cut = state.solver.cuts.cuts.get(index)?;
                if cut.denominator > 0 {
                    budget.spend(problem.model.networks[0].graph.vertices.len())?;
                    let weights = (0..problem.model.networks[0].graph.vertices.len())
                        .map(|vertex| {
                            i64::from(cut.weight_for_graph(
                                problem.model.networks[0].graph.graph_id.max(0),
                                vertex,
                            ))
                        })
                        .collect::<Vec<_>>();
                    if weights[problem.model.networks[0].graph.source] == 0
                        && weights[problem.model.networks[0].graph.sink] == 0
                    {
                        let audit = domain
                            .audit_weighted_upper_row(
                                &coefficients,
                                &weights,
                                cut.denominator as u64,
                                CoverRowLimits {
                                    max_work: budget.work,
                                    deadline: Some(deadline),
                                    ..Default::default()
                                },
                            )
                            .ok()?;
                        budget.spend(audit.work_used)?;
                        if audit.upper_bound.is_some_and(|maximum| {
                            (CoverRowRange {
                                minimum: maximum,
                                maximum,
                            })
                            .satisfies(master.senses[row], master.rhs[row])
                        }) {
                            keep[row] = true;
                            continue;
                        }
                    }
                }
            }
        }
        let audit = domain
            .audit_cover_row(
                &coefficients,
                CoverRowLimits {
                    max_states: 1 << 16,
                    max_work: budget.work,
                    deadline: Some(deadline),
                },
            )
            .ok()?;
        budget.spend(audit.work_used)?;
        keep[row] = audit
            .range
            .is_some_and(|range| range.satisfies(master.senses[row], master.rhs[row]));
    }
    let mut auxiliary = ClpMaster::new().ok()?;
    let mut compact = vec![None; keep.len()];
    for (row, &retain) in keep.iter().enumerate() {
        budget.spend(1)?;
        if retain {
            compact[row] = Some(auxiliary.rhs.len());
            auxiliary
                .add_row(master.senses[row], master.rhs[row], &[])
                .ok()?;
        }
    }
    // Original root routes must serve a customer. Empty source/sink paths
    // belong to the geometry collector but cannot satisfy vehicle branches.
    let mut position = 0;
    costs.retain(|_| {
        let retain = !direct[position];
        position += 1;
        retain
    });
    position = 0;
    entries.retain(|_| {
        let retain = !direct[position];
        position += 1;
        retain
    });
    for (cost, column) in costs.iter().zip(&mut entries) {
        budget.spend(cost.len() + column.len())?;
        column.retain_mut(|(row, _)| {
            if let Some(Some(mapped)) = compact.get(*row) {
                *row = *mapped;
                true
            } else {
                false
            }
        });
        // Floating aggregation affects LP dual quality only; the proof below
        // retains the raw costs and coefficients without cancellation.
        let mut aggregate = std::collections::BTreeMap::new();
        for &(row, value) in column.iter() {
            *aggregate.entry(row).or_insert(0.) += value;
        }
        let sparse = aggregate.into_iter().collect::<Vec<_>>();
        let cost = cost.iter().sum::<f64>();
        if !cost.is_finite() || sparse.iter().any(|&(_, a)| !a.is_finite()) {
            return None;
        }
        auxiliary.add_column(cost, &sparse).ok()?;
    }
    // Keep the auxiliary optimization feasible even on an infeasible child.
    // Slacks supply useful duals; they are excluded from the certified matrix.
    // Consequently their finite penalty is never an infeasibility argument.
    for row in 0..auxiliary.rhs.len() {
        for sign in [1., -1.] {
            budget.spend(1)?;
            if (sign > 0. && auxiliary.senses[row] == RowSense::LessEqual)
                || (sign < 0. && auxiliary.senses[row] == RowSense::GreaterEqual)
            {
                continue;
            }
            auxiliary.add_column(1e12, &[(row, sign)]).ok()?;
        }
    }
    budget.spend(0)?;
    if auxiliary
        .solve_limited(10_000, deadline.checked_duration_since(Instant::now())?)
        .ok()?
        != ClpSolveOutcome::Optimal
    {
        return None;
    }
    budget.spend(0)?;
    let mut offsets = Vec::new();
    for (route, quantity) in &problem.fixed_routes {
        for &id in &route.arcs {
            budget.store(1)?;
            offsets.push((
                problem.model.networks[0].graph.arcs[problem.arc_position(id)?].cost,
                f64::from(*quantity),
            ));
        }
    }
    let columns = costs
        .iter()
        .zip(&entries)
        .map(|(cost_terms, entries)| ProofColumn {
            cost_terms,
            entries,
        })
        .collect::<Vec<_>>();
    FiniteMaster {
        rhs: &auxiliary.rhs,
        senses: &auxiliary.senses,
        columns: &columns,
        offset_terms: &offsets,
        multiplicity_row: compact[map[0]]?,
    }
    .lower_bound_checked(&auxiliary.duals, || budget.spend(1))
}
