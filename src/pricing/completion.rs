//! Complete exact labels and restore the original arc domain.
use super::*;
impl<const D: usize> Solver<D> {
    pub fn complete_labels(&mut self, enforce_time_limit: bool) -> Result<()> {
        let exact = phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        };
        self.refresh_cuts()?;
        if self.engine.is_none() {
            self.engine = Some(self.new_engine(self.engine_controls(true), exact)?);
            self.prepared = Some(self.topology());
        }
        let engine = self.engine.as_mut().unwrap();
        let mut controls = self.controls.clone();
        if !enforce_time_limit {
            controls.time_limit = Duration::MAX;
        }
        engine.prepare_full_labels(&self.graph, &controls)
    }
    pub fn restore_arcs(&mut self) {
        self.engine = None;
        self.prepared = None;
        self.enumeration = None;
        self.enumeration_snapshot = None;
    }
}
