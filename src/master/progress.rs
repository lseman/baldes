//! Opt-in, per-solve column-generation progress. Formatting never affects solving.
use super::column_generation::{Iteration, Output};
use std::{
    io::{self, Write},
    time::Instant,
};

pub(super) struct ProgressLog {
    interval: Option<usize>,
    started: Instant,
    latest: Option<(Iteration, f64, f64)>,
    printed: Option<String>,
}

fn number(value: f64) -> String {
    if !value.is_finite() {
        return "--".into();
    }
    if value.abs() >= 1e9 || (value != 0. && value.abs() < 1e-4) {
        format!("{value:.4e}")
    } else {
        format!("{value:.5}")
    }
}

fn row(iteration: &Iteration, lower: f64, upper: f64, seconds: f64) -> String {
    let gap = if lower.is_finite() && upper.is_finite() {
        format!(
            "{:.2}",
            100. * (upper - lower).max(0.) / upper.abs().max(1.)
        )
    } else {
        "--".into()
    };
    format!(
        "{:>6} {:>13} {:>13} {:>13} {:>8} {:>11} {:>6} {:>7} {:>5} {:>8.2}",
        iteration.iteration,
        number(iteration.lp),
        number(lower),
        number(upper),
        gap,
        number(iteration.minimum),
        iteration.new_columns,
        iteration.columns,
        iteration.phase,
        seconds
    )
}

impl ProgressLog {
    pub(super) fn new(interval: Option<usize>) -> Self {
        if interval.is_some() {
            let mut out = io::stderr().lock();
            let _ = writeln!(out, "\n  BALDES | COLUMN GENERATION");
            let _ = writeln!(out, "{}", "─".repeat(99));
            let _ = writeln!(
                out,
                "{:>6} {:>13} {:>13} {:>13} {:>8} {:>11} {:>6} {:>7} {:>5} {:>8}",
                "Iter",
                "RMP obj",
                "Lower bound",
                "Incumbent",
                "Gap %",
                "Min RC",
                "+Cols",
                "Cols",
                "Phase",
                "Time(s)"
            );
            let _ = writeln!(out, "{}", "─".repeat(99));
        }
        Self {
            interval,
            started: Instant::now(),
            latest: None,
            printed: None,
        }
    }
    fn print_row(&mut self, line: String) {
        if self.printed.as_ref() != Some(&line) {
            let _ = writeln!(io::stderr().lock(), "{line}");
            self.printed = Some(line);
        }
    }
    pub(super) fn iteration(&mut self, iteration: &Iteration, bound: f64, incumbent: f64) {
        if let Some(interval) = self.interval {
            if iteration.iteration == 1 || iteration.iteration % interval == 0 {
                self.print_row(row(
                    iteration,
                    bound,
                    incumbent,
                    self.started.elapsed().as_secs_f64(),
                ));
            }
            self.latest = Some((iteration.clone(), bound, incumbent));
        }
    }
    pub(super) fn finish(&mut self, result: &crate::Result<Output>, incumbent: f64) {
        if self.interval.is_none() {
            return;
        }
        let seconds = self.started.elapsed().as_secs_f64();
        match result {
            Ok(output) => {
                if let Some((mut last, _, _)) = self.latest.take() {
                    last.iteration = output.iterations;
                    last.lp = output.lp;
                    last.columns = output.columns;
                    self.print_row(row(&last, output.bound, incumbent, seconds));
                }
                let mut out = io::stderr().lock();
                let _ = writeln!(out, "{}", "─".repeat(99));
                let _ = writeln!(
                    out,
                    "  CG {:?} | {} iterations | {} pricing calls | {} columns | {:.2}s\n",
                    output.status, output.iterations, output.pricing_calls, output.columns, seconds
                );
            }
            Err(error) => {
                let _ = writeln!(
                    io::stderr().lock(),
                    "  CG error after {seconds:.2}s: {error}\n"
                );
            }
        }
    }
}
