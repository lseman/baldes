//! Complete exact labels and restore the original arc domain.
use super::*;
impl<const D: usize> Solver<D> {
    pub fn complete_labels(&mut self, enforce_time_limit: bool) -> Result<()> {
        self.complete_labels_with_limit(if enforce_time_limit {
            self.controls.time_limit
        } else {
            Duration::MAX
        })
    }
    pub(super) fn complete_labels_with_limit(&mut self, limit: Duration) -> Result<()> {
        self.prepare_completion_engine()?;
        let engine = self.engine.as_mut().unwrap();
        let mut controls = self.controls.clone();
        controls.time_limit = limit;
        let result = engine.prepare_full_labels(&self.graph, &controls);
        if result.is_err() {
            // A failed completion can leave partially populated labels,
            // buckets and an error flag in the working engine.
            self.restore_arcs();
        }
        result
    }
    pub(super) fn prepare_completion_engine(&mut self) -> Result<()> {
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
        Ok(())
    }
    pub fn restore_arcs(&mut self) {
        self.raw_pricing = None;
        self.engine = None;
        self.prepared = None;
        self.enumeration = None;
        self.enumeration_snapshot = None;
    }
}
