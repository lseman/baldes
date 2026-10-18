//! Isolated HiGHS MIP solves from the complete current master matrix.
use super::*;
use crate::master::solvers::mip::{MipOptions, MipOutput, MipSolution, MipStatus};
use std::collections::HashSet;
use std::time::Instant;

extern "C" {
    fn Highs_getSizeofHighsInt(highs: *const c_void) -> HighsInt;
    fn Highs_setIntOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: HighsInt,
    ) -> HighsInt;
    #[allow(clippy::too_many_arguments)]
    fn Highs_passMip(
        highs: *mut c_void,
        cols: HighsInt,
        rows: HighsInt,
        nz: HighsInt,
        format: HighsInt,
        sense: HighsInt,
        offset: f64,
        cost: *const f64,
        lower: *const f64,
        upper: *const f64,
        row_lower: *const f64,
        row_upper: *const f64,
        starts: *const HighsInt,
        indices: *const HighsInt,
        values: *const f64,
        integrality: *const HighsInt,
    ) -> HighsInt;
    fn Highs_getIntInfoValue(
        highs: *const c_void,
        name: *const c_char,
        value: *mut HighsInt,
    ) -> HighsInt;
    fn Highs_getInt64InfoValue(
        highs: *const c_void,
        name: *const c_char,
        value: *mut i64,
    ) -> HighsInt;
    fn Highs_getDoubleInfoValue(
        highs: *const c_void,
        name: *const c_char,
        value: *mut f64,
    ) -> HighsInt;
}
fn count(n: usize) -> Result<HighsInt> {
    HighsInt::try_from(n).map_err(|_| invalid("HiGHS MIP index/count exceeds 32-bit build"))
}
/// Solve an independent HiGHS copy of any current LP master. Every physical
/// row, coefficient, cost and column upper bound is exported. Master columns
/// have lower bound zero by the LpMaster contract. Unlisted columns remain
/// continuous. The source model, solution and basis are untouched.
///
/// A finite objective cutoff filters improving witnesses and proof outcomes;
/// it is not installed as a row or a presumed incumbent. Infeasibility remains
/// distinguishable from having no solution that improves a supplied cutoff.
pub fn solve_integer<M: LpMaster>(
    source: &M,
    integer_columns: &[usize],
    options: &MipOptions,
) -> Result<MipOutput> {
    if options.objective_cutoff.is_nan() || options.objective_cutoff == f64::NEG_INFINITY {
        return Err(invalid("MIP cutoff must be finite or positive infinity"));
    }
    let offset = source.objective_offset();
    if !offset.is_finite() {
        return Err(invalid("finite MIP objective offset required"));
    }
    let (cols, rows) = (source.costs().len(), source.rhs().len());
    let (ncol, nrow, nodes) = (count(cols)?, count(rows)?, count(options.node_limit)?);
    if source.upper().len() != cols || source.senses().len() != rows {
        return Err(invalid("MIP source dimensions"));
    }
    if source.rhs().iter().any(|x| !x.is_finite()) {
        return Err(invalid("MIP source RHS"));
    }
    let mut integer_set = HashSet::new();
    let mut integrality = vec![0; cols];
    for &c in integer_columns {
        if c >= cols || !integer_set.insert(c) {
            return Err(invalid("MIP integer column index or duplicate"));
        }
        integrality[c] = 1;
    }
    let started = Instant::now();
    let mut output = MipOutput {
        status: MipStatus::Incomplete,
        solution: None,
        bound: f64::NEG_INFINITY,
        nodes: 0,
        simplex_iterations: 0,
        seconds: 0.,
        solver_status: 0,
    };
    if nodes == 0 || options.time_limit.is_zero() {
        output.status = if options.time_limit.is_zero() {
            MipStatus::TimeLimit
        } else {
            MipStatus::NodeLimit
        };
        return Ok(output);
    }
    if cols == 0 {
        let feasible = source
            .rhs()
            .iter()
            .enumerate()
            .all(|(r, &rhs)| match source.senses()[r] {
                RowSense::Equal => rhs.abs() <= 1e-7,
                RowSense::LessEqual => rhs >= -1e-7,
                RowSense::GreaterEqual => rhs <= 1e-7,
            });
        output.bound = if feasible { offset } else { f64::INFINITY };
        output.status = if !feasible {
            MipStatus::Infeasible
        } else if options.objective_cutoff <= offset {
            MipStatus::ObjectiveCutoff
        } else {
            output.solution = Some(MipSolution {
                objective: offset,
                values: vec![],
            });
            MipStatus::Optimal
        };
        output.seconds = started.elapsed().as_secs_f64();
        return Ok(output);
    }
    let model = HighsMaster::new()?;
    model.set_double("small_matrix_value", 1e-12)?;
    let h = model.highs.as_ptr();
    // SAFETY: live owned model; no model arrays are passed before checking ABI.
    if unsafe { Highs_getSizeofHighsInt(h) } != 4 {
        return Err(invalid("HiGHS requires a 32-bit HighsInt build"));
    }
    let infinity = model.infinity();
    let mut row_lower = Vec::with_capacity(rows);
    let mut row_upper = Vec::with_capacity(rows);
    for (r, &rhs) in source.rhs().iter().enumerate() {
        if !rhs.is_finite() {
            return Err(invalid("MIP source RHS"));
        }
        row_lower.push(if source.senses()[r] == RowSense::LessEqual {
            -infinity
        } else {
            rhs
        });
        row_upper.push(if source.senses()[r] == RowSense::GreaterEqual {
            infinity
        } else {
            rhs
        });
    }
    let mut starts = Vec::with_capacity(cols + 1);
    let mut indices = vec![];
    let mut values = vec![];
    for c in 0..cols {
        if !source.costs()[c].is_finite()
            || !source.upper()[c].is_finite()
            || source.upper()[c] < 0.
        {
            return Err(invalid("MIP source cost or upper bound"));
        }
        starts.push(count(indices.len())?);
        for (r, a) in source.column_entries(c)? {
            if r >= rows || !a.is_finite() {
                return Err(invalid("MIP source matrix"));
            }
            if a != 0. && a.abs() <= 1e-12 {
                return Err(invalid(
                    "MIP coefficient is below the HiGHS matrix retention threshold",
                ));
            }
            indices.push(count(r)?);
            values.push(a);
        }
    }
    starts.push(count(indices.len())?);
    let lower = vec![0.; cols];
    // SAFETY: checked dimensions and CSC pointers. HiGHS copies all model data.
    HighsMaster::check(
        unsafe {
            Highs_passMip(
                h,
                ncol,
                nrow,
                count(indices.len())?,
                1,
                1,
                offset,
                source.costs().as_ptr(),
                lower.as_ptr(),
                source.upper().as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
                integrality.as_ptr(),
            )
        },
        "passMip",
    )?;
    model.set_double("mip_rel_gap", 0.)?;
    model.set_double("mip_abs_gap", 0.)?;
    model.set_double("mip_feasibility_tolerance", 1e-7)?;
    // SAFETY: live handle and NUL-terminated static option name.
    HighsMaster::check(
        unsafe { Highs_setIntOptionValue(h, b"mip_max_nodes\0".as_ptr().cast(), nodes) },
        "mip_max_nodes",
    )?;
    let remaining = options.time_limit.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        output.status = MipStatus::TimeLimit;
        output.seconds = started.elapsed().as_secs_f64();
        return Ok(output);
    }
    model.set_double("time_limit", remaining.as_secs_f64())?;
    // SAFETY: live exclusively owned model. HiGHS controls its integer search.
    HighsMaster::check(unsafe { Highs_run(h) }, "integer run")?;
    output.solver_status = unsafe { Highs_getModelStatus(h) };
    let mut nodes_done = 0i64;
    let mut iterations = 0;
    let mut bound = f64::NEG_INFINITY;
    let mut primal_status = 0;
    // SAFETY: writable scalars of the exact types specified by highs_c_api.h.
    unsafe {
        if Highs_getInt64InfoValue(h, b"mip_node_count\0".as_ptr().cast(), &mut nodes_done) == 0 {
            output.nodes = usize::try_from(nodes_done.max(0)).unwrap_or(usize::MAX);
        }
        if Highs_getIntInfoValue(
            h,
            b"simplex_iteration_count\0".as_ptr().cast(),
            &mut iterations,
        ) == 0
        {
            output.simplex_iterations = iterations.max(0) as usize;
        }
        if Highs_getDoubleInfoValue(h, b"mip_dual_bound\0".as_ptr().cast(), &mut bound) == 0
            && !bound.is_nan()
        {
            output.bound = bound;
        }
        HighsMaster::check(
            Highs_getIntInfoValue(
                h,
                b"primal_solution_status\0".as_ptr().cast(),
                &mut primal_status,
            ),
            "primal status",
        )?;
    }
    let mut valid_witness = false;
    if primal_status == 2 {
        let mut solution = vec![0.; cols];
        let mut row_values = vec![0.; rows];
        // SAFETY: output buffers have the exact model dimensions; duals unused.
        HighsMaster::check(
            unsafe {
                Highs_getSolution(
                    h,
                    solution.as_mut_ptr(),
                    std::ptr::null_mut(),
                    row_values.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "integer solution",
        )?;
        for &c in integer_columns {
            if (solution[c] - solution[c].round()).abs() > 1e-7 {
                output.status = MipStatus::Incomplete;
                output.seconds = started.elapsed().as_secs_f64();
                return Ok(output);
            }
            solution[c] = solution[c].round();
        }
        let mut activity = vec![0.; rows];
        valid_witness = solution
            .iter()
            .enumerate()
            .all(|(c, &x)| x.is_finite() && x >= -1e-7 && x <= source.upper()[c] + 1e-7);
        for (c, &x) in solution.iter().enumerate() {
            for k in starts[c] as usize..starts[c + 1] as usize {
                activity[indices[k] as usize] += values[k] * x;
            }
        }
        valid_witness &= activity
            .iter()
            .enumerate()
            .all(|(r, &x)| x.is_finite() && x >= row_lower[r] - 1e-7 && x <= row_upper[r] + 1e-7);
        let objective: f64 = offset + source
            .costs()
            .iter()
            .zip(&solution)
            .map(|(c, x)| c * x)
            .sum::<f64>();
        valid_witness &= objective.is_finite();
        if valid_witness && objective < options.objective_cutoff {
            output.solution = Some(MipSolution {
                objective,
                values: solution,
            });
        }
        if valid_witness && output.solver_status == MODEL_OPTIMAL {
            output.bound = objective;
        }
    }
    output.status = match output.solver_status {
        MODEL_OPTIMAL if valid_witness => {
            if output.solution.is_some() {
                MipStatus::Optimal
            } else {
                MipStatus::ObjectiveCutoff
            }
        }
        MODEL_INFEASIBLE => {
            output.bound = f64::INFINITY;
            MipStatus::Infeasible
        }
        // An empty unconstrained zero-column model has a valid empty witness.
        6 if cols == 0 && rows == 0 => {
            output.bound = offset;
            if offset < options.objective_cutoff {
                output.solution = Some(MipSolution {
                    objective: offset,
                    values: vec![],
                });
                MipStatus::Optimal
            } else {
                MipStatus::ObjectiveCutoff
            }
        }
        13 => MipStatus::TimeLimit,
        14 if output.nodes >= options.node_limit => MipStatus::NodeLimit,
        _ => MipStatus::Incomplete,
    };
    if let Some(s) = &output.solution {
        output.bound = output.bound.min(s.objective);
    }
    output.seconds = started.elapsed().as_secs_f64();
    Ok(output)
}
