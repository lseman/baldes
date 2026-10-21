use super::SolveResult;
use std::time::Instant;

pub(super) struct ProgressLog {
    interval: Option<usize>,
    started: Instant,
    latest: Option<String>,
    printed_latest: bool,
}

impl ProgressLog {
    pub fn new(interval: Option<usize>, started: Instant) -> Self {
        if interval.is_some() {
            eprintln!("Column elimination: continuous route LP (fractional flows allowed)");
            eprintln!(
                "{:>6} {:>14} {:>9} {:>10} {:>9} {:>10} {:>9}",
                "Iter", "Lower bound", "Paths", "Infeasible", "Nodes", "Arcs", "Time(s)"
            );
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
            .map_or("--".into(), |v| format!("{v:.6}"));
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
            eprintln!("{row}");
        }
        self.latest = Some(row);
    }

    pub fn finish(&self, result: &SolveResult) {
        if self.interval.is_none() {
            return;
        }
        if !self.printed_latest {
            if let Some(row) = &self.latest {
                eprintln!("{row}");
            }
        }
        eprintln!(
            "CE finished: {:?}; iterations={} refinements={} nodes={} arcs={} time={:.3}s",
            result.status,
            result.iterations,
            result.refinements,
            result.nodes,
            result.arcs,
            self.started.elapsed().as_secs_f64()
        );
    }
}
