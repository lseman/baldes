use super::*;

impl<const D: usize> LabelArena<D> {
    pub(crate) fn expand(&mut self, ids: &[usize]) -> usize {
        let mut count = 0;
        match &mut self.records {
            Records::Ordinary(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].expanded) as usize;
                    v[id].expanded = true;
                }
            }
            Records::Extended(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Branched(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact2(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact4(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact8(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Full(v) => {
                for &id in ids {
                    count += (!v[id].expanded) as usize;
                    v[id].expanded = true;
                }
            }
        }
        count
    }

    pub(crate) fn unexpand_where(&mut self, mut predicate: impl FnMut(&[f64]) -> bool) {
        self.inspected.take();
        match &mut self.records {
            Records::Ordinary(v) => {
                for label in v {
                    if predicate(&label.resources) {
                        label.expanded = false;
                    }
                }
            }
            Records::Extended(v) => {
                for label in v {
                    if predicate(&label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Branched(v) => {
                for label in v {
                    if predicate(&label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact2(v) => {
                for label in v {
                    if predicate(&label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact4(v) => {
                for label in v {
                    if predicate(&label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact8(v) => {
                for label in v {
                    if predicate(&label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Full(v) => {
                for label in v {
                    if predicate(&label.resources) {
                        label.expanded = false;
                    }
                }
            }
        }
    }

    pub(super) fn full_mut(&mut self) -> &mut Vec<StoredLabel<D>> {
        if !matches!(self.records, Records::Full(_)) {
            let full = self
                .inspected
                .take()
                .unwrap_or_else(|| (0..self.len()).map(|id| self.view(id).to_owned()).collect());
            self.records = Records::Full(full);
        }
        match &mut self.records {
            Records::Full(v) => v,
            Records::Ordinary(_)
            | Records::Extended(_)
            | Records::Branched(_)
            | Records::Compact2(_)
            | Records::Compact4(_)
            | Records::Compact8(_) => unreachable!(),
        }
    }
}
