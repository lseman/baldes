use super::*;

impl<const D: usize> OrdinaryLabel<D> {
    #[inline(always)]
    pub(super) fn pack(label: &StoredLabel<D>, id: usize) -> Option<Self> {
        // Positive zero is required: preserve even the sign of zero in full
        // records instead of changing floating-point keys during conversion.
        if label.id != id
            || label.branch_state.is_some()
            || label.additional_cost.to_bits() != 0
            || label.half_potential.to_bits() != 0
            || label.cut_words.iter().any(|&w| w != 0)
        {
            return None;
        }
        Self::pack_metadata(label, id)
    }
    #[inline(always)]
    pub(super) fn pack_metadata(label: &StoredLabel<D>, id: usize) -> Option<Self> {
        if label.id != id {
            return None;
        }
        let optional = |value: Option<usize>| match value {
            None => Some(NONE),
            Some(value) => u32::try_from(value).ok().filter(|&v| v != NONE),
        };
        Some(Self {
            cost: label.cost,
            original_cost: label.original_cost,
            resources: label.resources.0,
            parent: optional(label.parent)?,
            arc: optional(label.arc)?,
            bucket_vertex: u32::try_from(label.bucket.vertex).ok()?,
            bucket: u32::try_from(label.bucket.bucket).ok()?,
            vertex: u32::try_from(label.vertex).ok()?,
            hops: u32::try_from(label.hops).ok()?,
            local_id: u32::try_from(label.local_id).ok()?,
            ng: label.ng,
            expanded: label.expanded,
        })
    }
    #[inline(always)]
    pub(super) fn view(&self, id: usize) -> LabelRef<'_, D> {
        LabelRef::Ordinary(self, id)
    }
    pub(super) fn unpack(self, id: usize) -> StoredLabel<D> {
        StoredLabel {
            id,
            local_id: self.local_id as u64,
            bucket: BucketId {
                vertex: self.bucket_vertex as usize,
                bucket: self.bucket as usize,
            },
            vertex: self.vertex as usize,
            cost: self.cost,
            original_cost: self.original_cost,
            resources: self.resources.into(),
            ng: self.ng,
            branch_state: None,
            parent: (self.parent != NONE).then_some(self.parent as usize),
            arc: (self.arc != NONE).then_some(self.arc as usize),
            hops: self.hops as usize,
            expanded: self.expanded,
            cut_words: EMPTY_WORDS,
            additional_cost: 0.,
            half_potential: 0.,
        }
    }
}

impl<const WORDS: usize, const D: usize> ExtendedLabel<WORDS, D> {
    #[inline(always)]
    pub(super) fn pack(label: &mut StoredLabel<D>, id: usize) -> Option<Self> {
        if label.cut_words.get(WORDS..)?.iter().any(|&w| w != 0) {
            return None;
        }
        Some(Self {
            base: OrdinaryLabel::pack_metadata(label, id)?,
            cut_words: label.cut_words.get(..WORDS)?.try_into().ok()?,
            additional_cost: label.additional_cost,
            half_potential: label.half_potential,
            branch_state: label.branch_state.take(),
        })
    }
    pub(super) fn unpack(&self, id: usize) -> StoredLabel<D> {
        let mut label = self.base.unpack(id);
        label.cut_words[..WORDS].copy_from_slice(&self.cut_words);
        label.additional_cost = self.additional_cost;
        label.half_potential = self.half_potential;
        label.branch_state = self.branch_state.clone();
        label
    }
}

impl<const D: usize> BranchedLabel<D> {
    #[inline(always)]
    pub(super) fn pack(label: &mut StoredLabel<D>, id: usize) -> Option<Self> {
        if label.additional_cost.to_bits() != 0
            || label.half_potential.to_bits() != 0
            || label.cut_words.iter().any(|&w| w != 0)
        {
            return None;
        }
        Some(Self {
            base: OrdinaryLabel::pack_metadata(label, id)?,
            branch_state: label.branch_state.take(),
        })
    }
    pub(super) fn unpack(&self, id: usize) -> StoredLabel<D> {
        let mut label = self.base.unpack(id);
        label.branch_state = self.branch_state.clone();
        label
    }
}
