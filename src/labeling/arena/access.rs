use super::*;

impl<const D: usize> LabelArena<D> {
    pub fn len(&self) -> usize {
        match &self.records {
            Records::Ordinary(v) => v.len(),
            Records::Extended(v) => v.len(),
            Records::Branched(v) => v.len(),
            Records::Compact2(v) => v.len(),
            Records::Compact4(v) => v.len(),
            Records::Compact8(v) => v.len(),
            Records::Auxiliary2(v) => v.len(),
            Records::Auxiliary4(v) => v.len(),
            Records::Auxiliary8(v) => v.len(),
            Records::AuxiliaryFull(v) => v.len(),
            Records::Full(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        match &self.records {
            Records::Ordinary(v) => v.capacity(),
            Records::Extended(v) => v.capacity(),
            Records::Branched(v) => v.capacity(),
            Records::Compact2(v) => v.capacity(),
            Records::Compact4(v) => v.capacity(),
            Records::Compact8(v) => v.capacity(),
            Records::Auxiliary2(v) => v.capacity(),
            Records::Auxiliary4(v) => v.capacity(),
            Records::Auxiliary8(v) => v.capacity(),
            Records::AuxiliaryFull(v) => v.capacity(),
            Records::Full(v) => v.capacity(),
        }
    }

    /// Heap capacity of retained records, including a lazily inspected copy.
    pub fn allocated_bytes(&self) -> usize {
        let records = match &self.records {
            Records::Ordinary(v) => v.capacity() * std::mem::size_of::<OrdinaryLabel<D>>(),
            Records::Extended(v) => {
                v.capacity() * std::mem::size_of::<ExtendedLabel<LIMITED_WORDS, D>>()
            }
            Records::Branched(v) => v.capacity() * std::mem::size_of::<BranchedLabel<D>>(),
            Records::Compact2(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<2, D>>(),
            Records::Compact4(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<4, D>>(),
            Records::Compact8(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<8, D>>(),
            Records::Auxiliary2(v) => v.capacity() * std::mem::size_of::<AuxiliaryLabel<2, D>>(),
            Records::Auxiliary4(v) => v.capacity() * std::mem::size_of::<AuxiliaryLabel<4, D>>(),
            Records::Auxiliary8(v) => v.capacity() * std::mem::size_of::<AuxiliaryLabel<8, D>>(),
            Records::AuxiliaryFull(v) => {
                v.capacity() * std::mem::size_of::<AuxiliaryLabel<LIMITED_WORDS, D>>()
            }
            Records::Full(v) => v.capacity() * std::mem::size_of::<StoredLabel<D>>(),
        };
        records
            + self
                .inspected
                .get()
                .map_or(0, |v| v.capacity() * std::mem::size_of::<StoredLabel<D>>())
    }

    pub(crate) fn kind(&self) -> RecordKind {
        match self.records {
            Records::Ordinary(_) => RecordKind::Ordinary,
            Records::Extended(_) => RecordKind::Extended,
            Records::Branched(_) => RecordKind::Branched,
            Records::Compact2(_) => RecordKind::Compact2,
            Records::Compact4(_) => RecordKind::Compact4,
            Records::Compact8(_) => RecordKind::Compact8,
            Records::Auxiliary2(_) => RecordKind::Auxiliary2,
            Records::Auxiliary4(_) => RecordKind::Auxiliary4,
            Records::Auxiliary8(_) => RecordKind::Auxiliary8,
            Records::AuxiliaryFull(_) => RecordKind::AuxiliaryFull,
            Records::Full(_) => RecordKind::Full,
        }
    }

    #[inline(always)]
    pub(crate) fn wide_record(&self, id: usize) -> &StoredLabel<D> {
        match &self.records {
            Records::Full(v) => &v[id],
            _ => unreachable!("unexpected label representation during a typed sweep"),
        }
    }

    #[inline(always)]
    pub(crate) fn words_for<R: ArenaRecord<D>>(&self, id: usize) -> &[u32] {
        if let Some(label) = R::get(self, id) {
            label.cut_words()
        } else {
            self.view(id).cut_words()
        }
    }

    #[inline(always)]
    pub(crate) fn branch_for<R: ArenaRecord<D>>(&self, id: usize) -> &BranchState {
        if let Some(label) = R::get(self, id) {
            label.branch_state()
        } else {
            self.view(id).branch_state()
        }
    }

    #[inline(always)]
    pub(crate) fn presence_for<R: ArenaRecord<D>>(
        &self,
        id: usize,
    ) -> &crate::cuts::soft_conflicts::LabelState {
        if let Some(label) = R::get(self, id) {
            label.presence_state()
        } else {
            self.view(id).presence_state()
        }
    }

    /// Borrow ordinary metadata directly in the sweep. Promotion remains safe:
    /// once an identifier needs wide storage, callers use the general view.
    #[inline(always)]
    pub(crate) fn ordinary_record(&self, id: usize) -> Option<&OrdinaryLabel<D>> {
        match &self.records {
            Records::Ordinary(v) => Some(&v[id]),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn extended_record(&self, id: usize) -> Option<&ExtendedLabel<LIMITED_WORDS, D>> {
        match &self.records {
            Records::Extended(v) => Some(&v[id]),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn branched_record(&self, id: usize) -> Option<&BranchedLabel<D>> {
        match &self.records {
            Records::Branched(v) => Some(&v[id]),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn view(&self, id: usize) -> LabelRef<'_, D> {
        match &self.records {
            Records::Ordinary(v) => v[id].view(id),
            Records::Extended(v) => LabelRef::Extended(&v[id], id),
            Records::Branched(v) => LabelRef::Branched(&v[id], id),
            Records::Compact2(v) => LabelRef::Compact2(&v[id], id),
            Records::Compact4(v) => LabelRef::Compact4(&v[id], id),
            Records::Compact8(v) => LabelRef::Compact8(&v[id], id),
            Records::Auxiliary2(v) => LabelRef::Auxiliary2(&v[id], id),
            Records::Auxiliary4(v) => LabelRef::Auxiliary4(&v[id], id),
            Records::Auxiliary8(v) => LabelRef::Auxiliary8(&v[id], id),
            Records::AuxiliaryFull(v) => LabelRef::AuxiliaryFull(&v[id], id),
            Records::Full(v) => (&v[id]).into(),
        }
    }
}
