use super::SolveResult;
use std::{
    io::{self, Write},
    time::Instant,
};

const TABLE_WIDTH: usize = 73;

pub(super) struct ProgressLog {
    interval: Option<usize>,
    started: Instant,
    latest: Option<String>,
    printed_latest: bool,
}

impl ProgressLog {
    pub fn new(interval: Option<usize>, started: Instant) -> Self {
        if interval.is_some() {
            let mut out = io::stderr().lock();
            let _ = writeln!(out, "\n  BALDES | COLUMN ELIMINATION");
            let _ = writeln!(out, "  Continuous route LP · fractional flows allowed");
            let _ = writeln!(out, "{}", "─".repeat(TABLE_WIDTH));
            let _ = writeln!(
                out,
                "{:>6} {:>14} {:>9} {:>10} {:>9} {:>10} {:>9}",
                "Iter", "Lower bound", "Paths", "Infeasible", "Nodes", "Arcs", "Time(s)"
            );
            let _ = writeln!(out, "{}", "─".repeat(TABLE_WIDTH));
        }
        Self {
            interval,
            started,
            latest: None,
            printed_latest: false,
        }
    }

    pub fn iteration(&mut self, result: &SolveResult, paths: usize, infeasible: usize) {
        let Some(interval) = self.interval else {
            return;
        };
        let bound = result
            .lower_bound
            .filter(|v| v.is_finite())
            .map_or("--".into(), |v| {
                if v.abs() >= 1e9 || (v != 0.0 && v.abs() < 1e-4) {
                    format!("{v:.4e}")
                } else {
                    format!("{v:.6}")
                }
            });
        let row = format!(
            "{:>6} {:>14} {:>9} {:>10} {:>9} {:>10} {:>9.2}",
            result.iterations,
            bound,
            paths,
            infeasible,
            result.nodes,
            result.arcs,
            self.started.elapsed().as_secs_f64()
        );
        self.printed_latest = result.iterations == 1 || result.iterations % interval == 0;
        if self.printed_latest {
            let _ = writeln!(io::stderr().lock(), "{row}");
        }
        self.latest = Some(row);
    }

    pub fn finish(&self, result: &SolveResult) {
        if self.interval.is_none() {
            return;
        }
        let mut out = io::stderr().lock();
        if !self.printed_latest {
            if let Some(row) = &self.latest {
                let _ = writeln!(out, "{row}");
            }
        }
        let _ = writeln!(out, "{}", "─".repeat(TABLE_WIDTH));
        let _ = writeln!(
            out,
            "  CE {:?} | {} iterations | {} refinements | {:.2}s",
            result.status,
            result.iterations,
            result.refinements,
            self.started.elapsed().as_secs_f64()
        );
        let _ = writeln!(
            out,
            "  Diagram: {} nodes | {} arcs\n",
            result.nodes, result.arcs
        );
    }
}
