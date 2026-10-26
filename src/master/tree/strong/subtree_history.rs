//! Completed committed-tree sizes, kept separate from proof bounds and trials.
struct Entry {
    parent: Option<usize>,
    depth: usize,
    pending: usize,
    size: usize,
    closed: bool,
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/strong/subtree_history/tests.rs"]
mod tests;
pub(in crate::master) struct SubtreeHistory {
    entries: Vec<Entry>,
    means: Vec<(usize, f64)>,
}
impl SubtreeHistory {
    pub fn new() -> Self {
        Self {
            entries: vec![Entry {
                parent: None,
                depth: 0,
                pending: 0,
                size: 1,
                closed: false,
            }],
            means: vec![],
        }
    }
    pub fn split(&mut self, parent: usize) -> [usize; 2] {
        let first = self.entries.len();
        let depth = self.entries[parent].depth + 1;
        self.entries[parent].pending = 2;
        for _ in 0..2 {
            self.entries.push(Entry {
                parent: Some(parent),
                depth,
                pending: 0,
                size: 1,
                closed: false,
            });
        }
        [first, first + 1]
    }
    pub fn close(&mut self, mut node: usize) {
        loop {
            let entry = &mut self.entries[node];
            if entry.closed || entry.pending != 0 {
                return;
            }
            entry.closed = true;
            let (size, depth, parent) = (entry.size, entry.depth, entry.parent);
            self.means.resize(self.means.len().max(depth + 1), (0, 0.));
            let (count, mean) = &mut self.means[depth];
            *mean = (*mean * *count as f64 + size as f64) / (*count + 1) as f64;
            *count += 1;
            let Some(parent) = parent else {
                return;
            };
            let entry = &mut self.entries[parent];
            entry.pending -= 1;
            entry.size = entry.size.saturating_add(size);
            node = parent;
        }
    }
    pub fn mean(&self, depth: usize) -> Option<f64> {
        self.means
            .get(depth)
            .filter(|(count, _)| *count > 0)
            .map(|(_, mean)| *mean)
    }
}
