//! Bounded optimization of explicit paired source controls on one fixed path.
//! LP cells cover the PWL/max source program. Numerical LP results supply
//! proposals/duals; source replay and finite-matrix bounds supply the proofs.
use crate::graph::coupled::{CoupledResource, PairedControl};
use crate::graph::{Graph, Window};
use crate::master::clp::{ClpMaster, ClpSolveOutcome};
use crate::{invalid, Result};
use std::time::{Duration, Instant};
pub(crate) mod affine;
mod linear;
mod program;
mod properties;
use linear::{solve_cell, CellResult};
use program::{cell, program, repair_arrivals};
pub(crate) use properties::integer_program_range;

#[derive(Clone, Copy, Debug)]
pub struct ControlSearchLimits {
    /// Explicit source-region limit. The default is unlimited; `max_work`, LP
    /// iterations and the optional deadline still bound every search.
    pub max_cells: usize,
    pub max_work: usize,
    pub max_lp_iterations: usize,
    pub deadline: Option<Instant>,
}
impl Default for ControlSearchLimits {
    fn default() -> Self {
        Self {
            max_cells: usize::MAX,
            max_work: 1_000_000,
            max_lp_iterations: 10_000,
            deadline: None,
        }
    }
}

/// `complete` means every source region was independently bounded or rejected,
/// and a feasible control was found (or every region was proved infeasible).
/// The lower bound is for signed terminal value: -Gamma when maximizing.
/// A complete result can have a nonzero certified primal/dual enclosure gap.
/// Incomplete results never assert infeasibility; they may retain a valid control.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ControlOptimization {
    pub maximize: bool,
    pub control: Option<PairedControl>,
    pub dependent: Option<f64>,
    pub lower_bound: Option<f64>,
    pub complete: bool,
    pub cells: usize,
    pub work_used: usize,
    pub lp_iterations: usize,
}
impl ControlOptimization {
    fn unknown(maximize: bool) -> Self {
        Self {
            maximize,
            control: None,
            dependent: None,
            lower_bound: None,
            complete: false,
            cells: 0,
            work_used: 0,
            lp_iterations: 0,
        }
    }
}

/// Query while retaining one caller-owned work/iteration budget across paths.
pub(crate) fn query<const D: usize>(
    graph: &Graph<D>,
    path: &[usize],
    resource: usize,
    maximize: bool,
    terminal: Option<Window>,
    limits: &mut ControlSearchLimits,
) -> Option<ControlOptimization> {
    let result =
        optimize_paired_control(graph, path, resource, maximize, terminal, *limits).ok()?;
    limits.max_work = limits.max_work.checked_sub(result.work_used)?;
    limits.max_lp_iterations = limits.max_lp_iterations.checked_sub(result.lp_iterations)?;
    Some(result)
}

pub(crate) struct TerminalDomain {
    pub lower: f64,
    pub upper: f64,
    pub minimum: PairedControl,
    pub maximum: PairedControl,
}
/// Bound all source controls, independently of objective weights and rows.
/// Inner None is a completely rejected domain; outer None is Unknown.
pub(crate) fn terminal_domain<const D: usize>(
    graph: &Graph<D>,
    path: &[usize],
    resource: usize,
    limits: &mut ControlSearchLimits,
) -> Option<Option<TerminalDomain>> {
    let low = query(graph, path, resource, false, None, limits)?;
    if !low.complete {
        return None;
    }
    let Some(minimum) = low.control else {
        return Some(None);
    };
    let high = query(graph, path, resource, true, None, limits)?;
    if !high.complete {
        return None;
    }
    let maximum = high.control?;
    let mut lower = low.lower_bound?;
    let mut upper = -high.lower_bound?;
    if let Some(terminal) = graph.coupled_kernel()?.paired_terminal_bounds(
        resource,
        path,
        &mut limits.max_work,
        limits.deadline,
    ) {
        lower = lower.max(terminal[0]);
        upper = upper.min(terminal[1]);
    }
    if let Some(value) =
        properties::invariant_value(graph, resource, path, &mut limits.max_work, limits.deadline)
    {
        lower = lower.max(value);
        upper = upper.min(value);
    }
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return None;
    }
    for value in [low.dependent?, high.dependent?] {
        if value < lower || value > upper {
            return None;
        }
    }
    Some(Some(TerminalDomain {
        lower,
        upper,
        minimum,
        maximum,
    }))
}

struct Budget {
    work: usize,
    iterations: usize,
    deadline: Option<Instant>,
}
impl Budget {
    fn take(&mut self, n: usize) -> Option<()> {
        if self.deadline.is_some_and(|end| Instant::now() >= end) {
            return None;
        }
        self.work = self.work.checked_sub(n)?;
        Some(())
    }
    fn solve(&mut self, master: &mut ClpMaster) -> Option<ClpSolveOutcome> {
        self.take(1)?;
        let time = self.deadline.map_or(Duration::from_secs(5), |end| {
            end.saturating_duration_since(Instant::now())
        });
        let before = master.simplex_iterations;
        let result = master
            .solve_limited(self.iterations, time.min(Duration::from_secs(5)))
            .ok()?;
        let used = master.simplex_iterations.saturating_sub(before).max(1);
        self.iterations = self.iterations.checked_sub(used)?;
        self.take(0)?;
        Some(result)
    }
}

fn exact_add(a: f64, b: f64) -> Option<f64> {
    let lo = crate::core::rounding::add_down(a, b)?;
    (lo == crate::core::rounding::add_up(a, b)?).then_some(lo)
}
fn exact_mul(a: f64, b: f64) -> Option<f64> {
    let lo = crate::core::rounding::multiply_down(a, b)?;
    (lo == -crate::core::rounding::multiply_down(-a, b)?).then_some(lo)
}
fn exact_sub(a: f64, b: f64) -> Option<f64> {
    exact_add(a, -b)
}

/// Optimize one paired descriptor's physical terminal value on a fixed path.
/// Domain cells and clamp branches are exhaustive when `complete` is true.
/// Other descriptors and physical/scalar/master constraints are not certified.
/// Unsupported source coefficient arithmetic/budgets return an incomplete result.
pub fn optimize_paired_control<const D: usize>(
    graph: &Graph<D>,
    path: &[usize],
    resource: usize,
    maximize: bool,
    terminal: Option<Window>,
    limits: ControlSearchLimits,
) -> Result<ControlOptimization> {
    if graph.front_program_domain() {
        return Err(invalid(
            "paired control optimization requires a source model",
        ));
    }
    let kernel = graph
        .coupled_kernel()
        .ok_or_else(|| invalid("paired control optimization requires resources"))?;
    let definition = kernel
        .definitions()
        .get(resource)
        .ok_or_else(|| invalid("unknown control resource"))?;
    if !matches!(
        definition,
        CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
    ) || path.iter().any(|&i| i >= graph.arcs.len())
    {
        return Err(invalid(
            "control optimization needs a paired descriptor and valid physical arc positions",
        ));
    }
    if let Some(window) = terminal {
        Window::new(window.lower, window.upper)?;
        if !window.lower.is_finite() || !window.upper.is_finite() {
            return Err(invalid("finite terminal control window required"));
        }
    }
    let mut budget = Budget {
        work: limits.max_work,
        iterations: limits.max_lp_iterations,
        deadline: limits.deadline,
    };
    let mut output = ControlOptimization::unknown(maximize);
    let Some(program) = program(definition, path, &mut budget) else {
        output.work_used = limits.max_work - budget.work;
        output.lp_iterations = limits.max_lp_iterations - budget.iterations;
        return Ok(output);
    };
    let radices = program.radices();
    let mut choices = vec![0; radices.len()];
    let mut covered = true;
    let mut finished = false;
    let mut minimum = f64::INFINITY;
    let mut best = f64::INFINITY;
    let mut feasible_cell = false;
    while output.cells < limits.max_cells && budget.take(1).is_some() {
        output.cells += 1;
        match cell(&program, &choices, terminal, &mut budget) {
            Ok(None) => {}
            Err(()) => {
                covered = false;
                break;
            }
            Ok(Some(cell)) => match solve_cell(&cell, maximize, &mut budget) {
                CellResult::Infeasible => {}
                CellResult::Unknown => {
                    covered = false;
                }
                CellResult::Bounded { bound, bases } => {
                    feasible_cell = true;
                    minimum = minimum.min(bound);
                    let mut candidates = vec![];
                    for base in bases {
                        candidates.push(base.clone());
                        if let Some(repaired) =
                            repair_arrivals(&program, &choices, &cell.bounds, base, &mut budget)
                        {
                            candidates.push(repaired);
                        }
                    }
                    for base in candidates {
                        let control = PairedControl { resource, base };
                        if let Some(certificate) = kernel.certify_paired_control(
                            path,
                            &control,
                            &mut budget.work,
                            budget.deadline,
                        )? {
                            let value = certificate.dependent[0];
                            let objective = if maximize { -value } else { value };
                            if certificate.certifies_value(value)
                                && terminal.map_or(true, |w| w.lower <= value && value <= w.upper)
                                && objective < best
                            {
                                best = objective;
                                output.dependent = Some(value);
                                output.control = Some(control);
                            }
                        }
                    }
                }
            },
        }
        let mut carry = true;
        for (choice, &radix) in choices.iter_mut().zip(&radices).rev() {
            *choice += 1;
            if *choice < radix {
                carry = false;
                break;
            }
            *choice = 0;
        }
        if carry {
            finished = true;
            break;
        }
    }
    covered &= budget.take(0).is_some();
    // A source witness also guards consistency of the compiled region bounds.
    if feasible_cell && output.control.is_some() && minimum > best {
        covered = false;
    }
    output.complete = finished && covered && (!feasible_cell || output.control.is_some());
    if finished && covered && feasible_cell {
        output.lower_bound = Some(minimum);
    }
    output.work_used = limits.max_work - budget.work;
    output.lp_iterations = limits.max_lp_iterations - budget.iterations;
    Ok(output)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/controls_tests.rs"]
mod tests;
