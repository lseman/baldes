//! Install original-cost phase-I columns without exposing auxiliary objectives.
use super::controller::apply_column_cap;
use super::*;

pub(super) struct Columns<'a> {
    pub loaded: &'a mut Vec<Option<Column>>,
    pub seen: &'a mut HashSet<String>,
    pub pool: &'a mut Vec<Column>,
    pub pool_seen: &'a mut HashSet<String>,
}

pub(super) fn install<M: LpMaster, C: Controller<M>>(
    master: &mut M,
    controller: &C,
    recovery: FeasibilityRecovery,
    out: &mut Output,
    columns: Columns<'_>,
) -> Result<bool> {
    let Columns {
        loaded,
        seen,
        pool,
        pool_seen,
    } = columns;
    let report = recovery.report;
    if report.status == FeasibilityStatus::Infeasible {
        if out.bound_mode == BoundMode::Certified && !report.certified_infeasible {
            return Err(invalid("uncertified phase-I infeasibility"));
        }
        out.status = Status::Infeasible;
        out.lp = f64::INFINITY;
        out.bound = f64::INFINITY;
        if report.certified_infeasible {
            out.certified_bound = f64::INFINITY;
        }
        out.numerical_bound = f64::INFINITY;
        out.iterations += report.iterations;
        out.pricing_calls += report.pricing_calls;
        out.feasibility.push(report);
        return Ok(false);
    }
    let mut next = master
        .try_clone_for_transaction()?
        .ok_or_else(|| invalid("phase-I installation requires an independent master"))?;
    let mut present = seen.clone();
    let mut added = vec![];
    for column in recovery.columns {
        if !column.acceptable
            || !column.cost.is_finite()
            || column
                .entries
                .iter()
                .any(|&(r, a)| r >= next.rhs().len() || !a.is_finite())
        {
            return Err(invalid("invalid recovered phase-I column"));
        }
        if !present.insert(column.signature()) {
            continue;
        }
        let index = next.add_column(column.cost, &column.entries)?;
        apply_column_cap(&mut next, controller, index, column.tag)?;
        added.push(column);
    }
    if report.status == FeasibilityStatus::Feasible {
        // With the recovered real feasible point, artificials must no longer
        // compete with expensive real routes under the restored objective.
        for &column in &out.artificials {
            next.set_upper(column, 0.)?;
        }
    }
    *master = next;
    let inserted = added.len();
    for column in added {
        let identity = column.signature();
        seen.insert(identity.clone());
        if pool_seen.insert(identity) {
            pool.push(column.clone());
        }
        out.column_tags.push(column.tag);
        out.inserted_tags.push(column.tag);
        loaded.push(Some(column));
    }
    out.columns = master.costs().len();
    out.iterations += report.iterations;
    out.pricing_calls += report.pricing_calls;
    out.status = match report.status {
        FeasibilityStatus::TimeLimit => Status::TimeLimit,
        FeasibilityStatus::IterationLimit => Status::IterationLimit,
        _ => Status::Stalled,
    };
    let resume = report.status == FeasibilityStatus::Feasible
        || (report.status == FeasibilityStatus::Stalled && inserted > 0);
    out.feasibility.push(report);
    Ok(resume)
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/cg/feasibility_tests.rs"]
mod tests;
