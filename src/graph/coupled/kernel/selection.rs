//! Relaxed selection memory: check-only arcs that the master narrows.
use super::*;

impl CoupledKernel {
    /// Whether any selection resource starts check-only.
    pub(crate) fn selection_relaxed(&self) -> bool {
        self.selection_memory.iter().any(Option::is_some)
    }

    /// The exact selection rule: every resource keeps a value along `arcs`.
    pub(crate) fn selection_proper(&self, arcs: &[usize]) -> bool {
        self.selection_masks().all(|(initial, allowed, _)| {
            arcs.iter()
                .try_fold(initial, |domain, &arc| {
                    let next = domain & allowed.get(arc)?;
                    (next != 0).then_some(next)
                })
                .is_some()
        })
    }

    /// Narrow the domain on the arcs of each relaxed resource's violating
    /// windows in `arcs`. A window [i, j] is shortest: its masks intersect
    /// to empty and no proper subwindow does. Remembering all of i..=j makes
    /// forward and backward extension reject it, and a join splitting it
    /// intersects to empty, so the relaxation then excludes this route.
    /// Returns whether any arc started narrowing.
    pub(crate) fn strengthen_selection(&mut self, arcs: &[usize]) -> bool {
        let windows: Vec<(usize, Vec<usize>)> = self
            .selection_masks()
            .enumerate()
            .filter(|(_, (_, _, memory))| memory.is_some())
            .map(|(r, (initial, allowed, _))| (r, violating_arcs(initial, allowed, arcs)))
            .collect();
        let mut changed = false;
        for (r, remember) in windows {
            let memory = self.selection_memory[r].as_mut().expect("relaxed resource");
            for arc in remember {
                changed |= !std::mem::replace(&mut memory[arc], true);
            }
        }
        if changed {
            // Caches keyed by this kernel must not survive the new relaxation.
            self.identity = std::sync::Arc::new(());
        }
        changed
    }

    fn selection_masks(&self) -> impl Iterator<Item = (u64, &[u64], Option<&Vec<bool>>)> {
        self.definitions
            .iter()
            .filter_map(|r| match r {
                CoupledResource::Selection {
                    initial, allowed, ..
                } => Some((*initial, allowed.as_slice())),
                _ => None,
            })
            .zip(&self.selection_memory)
            .map(|((initial, allowed), memory)| (initial, allowed, memory.as_ref()))
    }
}

/// Arcs to remember: every arc of each shortest empty window [i, j].
fn violating_arcs(initial: u64, allowed: &[u64], arcs: &[usize]) -> Vec<usize> {
    let mut remember = vec![];
    for end in 0..arcs.len() {
        let Some(&mask) = allowed.get(arcs[end]) else {
            return remember;
        };
        // The latest start whose window up to `end` is already empty.
        let mut domain = initial & mask;
        let mut start = end;
        while domain != 0 && start > 0 {
            start -= 1;
            domain &= allowed[arcs[start]];
        }
        if domain != 0 {
            continue;
        }
        // Shortest only if the window without its last arc is nonempty.
        let shorter = arcs[start..end]
            .iter()
            .fold(initial, |d, &arc| d & allowed[arc]);
        if start < end && shorter != 0 {
            remember.extend_from_slice(&arcs[start..=end]);
        }
    }
    remember
}
