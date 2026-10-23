//! Opt-in, per-solve column-generation progress. Formatting never affects solving.
use super::cg::{Iteration, Output};
use std::{
    io::{self, Write},
    time::Instant,
};

pub(super) struct ProgressLog {
    interval: Option<usize>,
    started: Instant,
    latest: Option<(Iteration, f64, f64)>,
    printed: Option<String>,
    cuts: Vec<(&'static str, usize)>,
    bucket_steps: Vec<f64>,
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

fn steps(values: &[f64]) -> String {
    if values.is_empty() {
        "--".into()
    } else {
        values
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join("/")
    }
}

fn row(
    iteration: &Iteration,
    lower: f64,
    upper: f64,
    seconds: f64,
    cuts: &[(&str, usize)],
    bucket_steps: &[f64],
) -> String {
    let gap = if lower.is_finite() && upper.is_finite() {
        format!(
            "{:.2}",
            100. * (upper - lower).max(0.) / upper.abs().max(1.)
        )
    } else {
        "--".into()
    };
    let mut line = format!(
        "{:>6} {:>13} {:>13} {:>13} {:>8} {:>11} {:>6} {:>7}",
        iteration.iteration,
        number(iteration.lp),
        number(lower),
        number(upper),
        gap,
        number(iteration.minimum),
        iteration.new_columns,
        iteration.columns
    );
    for (_, count) in cuts {
        line.push_str(&format!(" {count:>8}"));
    }
    line.push_str(&format!(
        " {:>15} {:>5} {:>8.2}",
        steps(bucket_steps),
        iteration.phase,
        seconds
    ));
    line
}

fn header(cuts: &[(&str, usize)]) -> String {
    let mut line = format!(
        "{:>6} {:>13} {:>13} {:>13} {:>8} {:>11} {:>6} {:>7}",
        "Iter", "RMP obj", "Lower bound", "Incumbent", "Gap %", "Min RC", "+Cols", "Cols"
    );
    for (name, _) in cuts {
        line.push_str(&format!(" {name:>8}"));
    }
    line.push_str(&format!(
        " {:>15} {:>5} {:>8}",
        "Bucket step", "Phase", "Time(s)"
    ));
    line
}

impl ProgressLog {
    pub(super) fn new(interval: Option<usize>, cuts: Vec<(&'static str, usize)>) -> Self {
        if interval.is_some() {
            let mut out = io::stderr().lock();
            let _ = crate::logging::banner(&mut out, "COLUMN GENERATION");
            let _ = writeln!(out, "{}", "─".repeat(115 + cuts.len() * 9));
            let _ = writeln!(out, "{}", header(&cuts));
            let _ = writeln!(out, "{}", "─".repeat(115 + cuts.len() * 9));
        }
        Self {
            interval,
            started: Instant::now(),
            latest: None,
            printed: None,
            cuts,
            bucket_steps: vec![],
        }
    }
    fn print_row(&mut self, line: String) {
        if self.printed.as_ref() != Some(&line) {
            let _ = writeln!(io::stderr().lock(), "{line}");
            self.printed = Some(line);
        }
    }
    fn update_bucket_steps(&mut self, bucket_steps: Vec<f64>) {
        if !self.bucket_steps.is_empty() && self.bucket_steps != bucket_steps {
            let _ = writeln!(
                io::stderr().lock(),
                "  Bucket step updated: {} -> {}",
                steps(&self.bucket_steps),
                steps(&bucket_steps)
            );
        }
        self.bucket_steps = bucket_steps;
    }
    pub(super) fn iteration(
        &mut self,
        iteration: &Iteration,
        bound: f64,
        incumbent: f64,
        cuts: Vec<(&'static str, usize)>,
        bucket_steps: Vec<f64>,
    ) {
        if let Some(interval) = self.interval {
            self.cuts = cuts;
            self.update_bucket_steps(bucket_steps);
            if iteration.iteration == 1 || iteration.iteration % interval == 0 {
                self.print_row(row(
                    iteration,
                    bound,
                    incumbent,
                    self.started.elapsed().as_secs_f64(),
                    &self.cuts,
                    &self.bucket_steps,
                ));
            }
            self.latest = Some((iteration.clone(), bound, incumbent));
        }
    }
    pub(super) fn finish(
        &mut self,
        result: &crate::Result<Output>,
        incumbent: f64,
        cuts: Vec<(&'static str, usize)>,
        bucket_steps: Vec<f64>,
    ) {
        if self.interval.is_none() {
            return;
        }
        self.cuts = cuts;
        self.update_bucket_steps(bucket_steps);
        let seconds = self.started.elapsed().as_secs_f64();
        match result {
            Ok(output) => {
                if let Some((mut last, _, _)) = self.latest.take() {
                    last.iteration = output.iterations;
                    last.lp = output.lp;
                    last.columns = output.columns;
                    self.print_row(row(
                        &last,
                        output.bound,
                        incumbent,
                        seconds,
                        &self.cuts,
                        &self.bucket_steps,
                    ));
                }
                let mut out = io::stderr().lock();
                let _ = writeln!(out, "{}", "─".repeat(115 + self.cuts.len() * 9));
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
