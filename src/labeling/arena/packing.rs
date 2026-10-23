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
        if label.id != id || label.presence_state.is_some() || label.coupled_state.is_some() {
            return None;
        }
        Self::pack_fields(label, label.local_id)
    }
    /// Header-only packing is shared with auxiliary records and retained
    /// borrowed labels. Callers validate their own payload and ID contract.
    #[inline(always)]
    pub(super) fn pack_fields(label: &StoredLabel<D>, local_id: u64) -> Option<Self> {
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
            local_id: u32::try_from(local_id).ok()?,
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
            presence_state: None,
            coupled_state: None,
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

impl<const D: usize> OrdinaryLabel<D> {
    #[inline(always)]
    pub(super) fn pack_retained(label: &StoredLabel<D>, local_id: u64) -> Option<Self> {
        if label.branch_state.is_some()
            || label.presence_state.is_some()
            || label.coupled_state.is_some()
            || label.additional_cost.to_bits() != 0
            || label.half_potential.to_bits() != 0
            || label.cut_words.iter().any(|&w| w != 0)
        {
            return None;
        }
        Self::pack_fields(label, local_id)
    }
}
impl<const WORDS: usize, const D: usize> ExtendedLabel<WORDS, D> {
    #[inline(always)]
    pub(super) fn pack_retained(label: &StoredLabel<D>, local_id: u64) -> Option<Self> {
        if label.presence_state.is_some()
            || label.coupled_state.is_some()
            || label.cut_words.get(WORDS..)?.iter().any(|&w| w != 0)
        {
            return None;
        }
        Some(Self {
            base: OrdinaryLabel::pack_fields(label, local_id)?,
            cut_words: label.cut_words.get(..WORDS)?.try_into().ok()?,
            additional_cost: label.additional_cost,
            half_potential: label.half_potential,
            branch_state: label.branch_state.clone(),
        })
    }
}
impl<const D: usize> BranchedLabel<D> {
    #[inline(always)]
    pub(super) fn pack_retained(label: &StoredLabel<D>, local_id: u64) -> Option<Self> {
        if label.presence_state.is_some()
            || label.coupled_state.is_some()
            || label.additional_cost.to_bits() != 0
            || label.half_potential.to_bits() != 0
            || label.cut_words.iter().any(|&w| w != 0)
        {
            return None;
        }
        Some(Self {
            base: OrdinaryLabel::pack_fields(label, local_id)?,
            branch_state: label.branch_state.clone(),
        })
    }
}
impl<const WORDS: usize, const D: usize> AuxiliaryLabel<WORDS, D> {
    #[inline(always)]
    pub(super) fn pack(label: &mut StoredLabel<D>, id: usize) -> Option<Self> {
        if label.id != id || label.cut_words.get(WORDS..)?.iter().any(|&w| w != 0) {
            return None;
        }
        let base = OrdinaryLabel::pack_fields(label, label.local_id)?;
        let cut_words = label.cut_words.get(..WORDS)?.try_into().ok()?;
        Some(Self {
            base,
            cut_words,
            additional_cost: label.additional_cost,
            half_potential: label.half_potential,
            branch_state: label.branch_state.take(),
            presence_state: label.presence_state.take(),
            coupled_state: label.coupled_state.take(),
        })
    }
    #[inline(always)]
    pub(super) fn pack_retained(label: &StoredLabel<D>, local_id: u64) -> Option<Self> {
        if label.cut_words.get(WORDS..)?.iter().any(|&w| w != 0) {
            return None;
        }
        Some(Self {
            base: OrdinaryLabel::pack_fields(label, local_id)?,
            cut_words: label.cut_words.get(..WORDS)?.try_into().ok()?,
            additional_cost: label.additional_cost,
            half_potential: label.half_potential,
            branch_state: label.branch_state.clone(),
            presence_state: label.presence_state.clone(),
            coupled_state: label.coupled_state.clone(),
        })
    }
    pub(super) fn unpack(&self, id: usize) -> StoredLabel<D> {
        let mut label = self.base.unpack(id);
        label.cut_words[..WORDS].copy_from_slice(&self.cut_words);
        label.additional_cost = self.additional_cost;
        label.half_potential = self.half_potential;
        label.branch_state = self.branch_state.clone();
        label.presence_state = self.presence_state.clone();
        label.coupled_state = self.coupled_state.clone();
        label
    }
}
