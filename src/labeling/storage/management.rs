use super::*;

impl<const D: usize> LabelStorage<D> {
    pub fn new(counts: impl IntoIterator<Item = usize>) -> Self {
        let buckets: Vec<Vec<_>> = counts
            .into_iter()
            .map(|n| {
                (0..n)
                    .map(|_| LabelBucket {
                        lower_cost: 1e12,
                        knapsack_lower_cost: 1e12,
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        let crossing = vec![vec![]; buckets.len()];
        Self {
            labels: LabelArena::default(),
            hot: vec![],
            buckets,
            crossing,
            root: None,
        }
    }

    /// `*self = Self::new(counts)`, keeping the allocations when the bucket
    /// shape is unchanged; the arena and buckets regrow every pricing call.
    pub fn reset(&mut self, counts: impl IntoIterator<Item = usize>) {
        let counts: Vec<usize> = counts.into_iter().collect();
        if !counts.iter().copied().eq(self.buckets.iter().map(Vec::len)) {
            *self = Self::new(counts);
            return;
        }
        self.labels.clear();
        self.hot.clear();
        for bucket in self.buckets.iter_mut().flatten() {
            bucket.active.clear();
            bucket.hot.truncate(0);
            bucket.archived.clear();
            bucket.next_id = 0;
            bucket.lower_cost = 1e12;
            bucket.knapsack_lower_cost = 1e12;
        }
        self.crossing.iter_mut().for_each(Vec::clear);
        self.root = None;
    }

    pub fn retain(&mut self, mut label: StoredLabel<D>) -> usize {
        let id = self.labels.len();
        label.id = id;
        self.hot.push(HotLabel::of(&label));
        self.labels.push(label);
        id
    }

    #[inline(never)]
    pub(crate) fn retain_for<R: ArenaRecord<D>>(&mut self, mut label: StoredLabel<D>) -> usize {
        let id = self.labels.len();
        label.id = id;
        self.hot.push(HotLabel::of(&label));
        self.labels.push_for::<R>(label);
        id
    }
}

impl<const D: usize> LabelStorage<D> {
    #[inline(never)]
    pub(crate) fn retain_borrowed_for<R: ArenaRecord<D>>(
        &mut self,
        label: &StoredLabel<D>,
        local_id: u64,
    ) -> usize {
        let id = self.labels.len();
        self.hot.push(HotLabel::of(label));
        self.labels.push_borrowed_for::<R>(label, local_id);
        id
    }
}
