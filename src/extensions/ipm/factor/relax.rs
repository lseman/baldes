//! Bounded merging of adjacent child/parent fronts. The permutation stays fixed.
use super::*;

impl Symbolic {
    pub(crate) fn relax(&mut self, boundary: usize) {
        // Appended-column plans can already contain artificial fill. Do not
        // replenish the budget when reusing or cloning a relaxed plan.
        if self.relaxed {
            return;
        }
        let n = self.order.len();
        if n < 64 || self.fronts.len() < 16 {
            return;
        }
        if !self.fronts.windows(2).any(|pair| {
            (boundary == n || pair[0].lo >= boundary)
                && pair[1].end - pair[0].lo <= 64
                && pair[0].updates.first() == Some(&pair[1].lo)
        }) {
            return;
        }
        let entries = |f: &Front| {
            let width = f.end - f.lo;
            width * (width - 1) / 2 + width * f.updates.len()
        };
        let work = |f: &Front| {
            let width = f.end - f.lo;
            let size = width + f.updates.len();
            size.saturating_mul(size).saturating_mul(width)
        };
        // Bound total artificial factor entries, rather than compounding a
        // percentage allowance at each successive merge.
        let budget = self.fronts.iter().map(entries).sum::<usize>() / 8;
        let mut added = 0usize;
        let mut merged: Vec<Front> = Vec::with_capacity(self.fronts.len());
        let mut changed = false;
        for front in &self.fronts {
            if let Some(child) = merged.last_mut() {
                // Preserve appended-column reuse's independent primal fronts.
                // A normal-equation plan has boundary == n and may merge all.
                let eligible = (boundary == n || child.lo >= boundary)
                    && child.end == front.lo
                    && front.end - child.lo <= 64
                    && child.updates.first() == Some(&front.lo);
                if eligible {
                    let mut updates: Vec<_> = child
                        .updates
                        .iter()
                        .chain(&front.updates)
                        .copied()
                        .filter(|&i| i >= front.end)
                        .collect();
                    updates.sort_unstable();
                    updates.dedup();
                    let candidate = Front {
                        lo: child.lo,
                        end: front.end,
                        updates,
                        children: Vec::new(),
                    };
                    let extra = entries(&candidate).saturating_sub(entries(child) + entries(front));
                    let before = work(child).saturating_add(work(front));
                    if extra <= budget - added
                        && work(&candidate) <= before.saturating_add(before / 2)
                    {
                        added += extra;
                        *child = candidate;
                        changed = true;
                        continue;
                    }
                }
            }
            let mut front = front.clone();
            front.children.clear();
            merged.push(front);
        }
        if !changed {
            return;
        }
        let mut owner = vec![0; n];
        for (s, front) in merged.iter().enumerate() {
            owner[front.lo..front.end].fill(s);
        }
        for s in 0..merged.len() {
            if let Some(&column) = merged[s].updates.first() {
                merged[owner[column]].children.push(s);
            }
        }
        self.levels = Self::tree_levels(&merged);
        self.fronts = merged;
        self.relaxed = true;
        self.assembly = Arc::new(Vec::new());
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/ipm/factor/relax_tests.rs"]
mod tests;
