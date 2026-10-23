use super::*;

impl<const D: usize> LabelArena<D> {
    pub fn clear(&mut self) {
        self.inspected.take();
        match &mut self.records {
            Records::Ordinary(v) => v.clear(),
            Records::Extended(v) => v.clear(),
            Records::Branched(v) => v.clear(),
            Records::Compact2(v) => v.clear(),
            Records::Compact4(v) => v.clear(),
            Records::Compact8(v) => v.clear(),
            Records::Auxiliary2(v) => v.clear(),
            Records::Auxiliary4(v) => v.clear(),
            Records::Auxiliary8(v) => v.clear(),
            Records::AuxiliaryFull(v) => v.clear(),
            Records::Full(v) => v.clear(),
        }
    }

    pub(crate) fn set_word_limit(&mut self, words: usize) {
        assert!(words <= LIMITED_WORDS);
        self.word_limit = words;
    }

    #[cfg(test)]
    pub(crate) fn set_mode(&mut self, ordinary: bool, branch_only: bool) {
        self.set_resource_mode(ordinary, branch_only, false);
    }

    pub(crate) fn set_resource_mode(&mut self, ordinary: bool, branch_only: bool, auxiliary: bool) {
        assert!(
            self.is_empty(),
            "label representation changes require an empty arena"
        );
        assert!(!(ordinary && branch_only));
        self.inspected.take();
        let target = if auxiliary {
            if self.word_limit <= 2 {
                RecordKind::Auxiliary2
            } else if self.word_limit <= 4 {
                RecordKind::Auxiliary4
            } else if self.word_limit <= 8 {
                RecordKind::Auxiliary8
            } else {
                RecordKind::AuxiliaryFull
            }
        } else if ordinary {
            RecordKind::Ordinary
        } else if branch_only {
            RecordKind::Branched
        } else if self.word_limit <= 2 {
            RecordKind::Compact2
        } else if self.word_limit <= 4 {
            RecordKind::Compact4
        } else if self.word_limit <= 8 {
            RecordKind::Compact8
        } else {
            RecordKind::Extended
        };
        if self.kind() == target {
            return;
        }
        self.records = match target {
            RecordKind::Ordinary => Records::Ordinary(vec![]),
            RecordKind::Branched => Records::Branched(vec![]),
            RecordKind::Compact2 => Records::Compact2(vec![]),
            RecordKind::Compact4 => Records::Compact4(vec![]),
            RecordKind::Compact8 => Records::Compact8(vec![]),
            RecordKind::Auxiliary2 => Records::Auxiliary2(vec![]),
            RecordKind::Auxiliary4 => Records::Auxiliary4(vec![]),
            RecordKind::Auxiliary8 => Records::Auxiliary8(vec![]),
            RecordKind::AuxiliaryFull => Records::AuxiliaryFull(vec![]),
            _ => Records::Extended(vec![]),
        };
    }

    #[inline(always)]
    pub fn push(&mut self, mut label: StoredLabel<D>) {
        match &mut self.records {
            Records::Ordinary(v) => {
                self.inspected.take();
                if let Some(compact) = OrdinaryLabel::pack(&label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Extended(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<LIMITED_WORDS, D>::pack(&mut label, v.len())
                {
                    v.push(compact);
                    return;
                }
            }
            Records::Branched(v) => {
                self.inspected.take();
                if let Some(compact) = BranchedLabel::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Compact2(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<2, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Compact4(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<4, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Compact8(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<8, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Auxiliary2(v) => {
                self.inspected.take();
                if let Some(compact) = AuxiliaryLabel::<2, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Auxiliary4(v) => {
                self.inspected.take();
                if let Some(compact) = AuxiliaryLabel::<4, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Auxiliary8(v) => {
                self.inspected.take();
                if let Some(compact) = AuxiliaryLabel::<8, D>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::AuxiliaryFull(v) => {
                self.inspected.take();
                if let Some(compact) = AuxiliaryLabel::<LIMITED_WORDS, D>::pack(&mut label, v.len())
                {
                    v.push(compact);
                    return;
                }
            }
            Records::Full(v) => {
                v.push(label);
                return;
            }
        }
        // Preserve all state and identifiers instead of truncating or dropping
        // a payload that cannot be encoded by the selected compact record.
        self.full_mut().push(label);
    }

    #[inline(always)]
    pub(crate) fn push_for<R: ArenaRecord<D>>(&mut self, mut label: StoredLabel<D>) {
        self.inspected.take();
        if !R::try_push(self, &mut label) {
            R::push_fallback(self, label);
        }
    }

    #[inline(never)]
    pub(super) fn push_wide(&mut self, label: StoredLabel<D>) {
        self.full_mut().push(label);
    }
}

impl<const D: usize> LabelArena<D> {
    #[inline(always)]
    pub(crate) fn push_borrowed_for<R: ArenaRecord<D>>(
        &mut self,
        label: &StoredLabel<D>,
        local_id: u64,
    ) {
        self.inspected.take();
        if !R::try_push_borrowed(self, label, local_id) {
            // Only an actual wide fallback materializes a full temporary.
            let mut owned = label.clone();
            owned.id = self.len();
            owned.local_id = local_id;
            R::push_fallback(self, owned);
        }
    }
    #[inline(always)]
    pub(super) fn try_push_borrowed_current(
        &mut self,
        label: &StoredLabel<D>,
        local_id: u64,
    ) -> bool {
        match self.kind() {
            RecordKind::Ordinary => {
                <OrdinaryLabel<D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Branched => {
                <BranchedLabel<D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Extended => {
                <ExtendedLabel<LIMITED_WORDS, D> as ArenaRecord<D>>::try_push_borrowed(
                    self, label, local_id,
                )
            }
            RecordKind::Compact2 => {
                <ExtendedLabel<2, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Compact4 => {
                <ExtendedLabel<4, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Compact8 => {
                <ExtendedLabel<8, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Auxiliary2 => {
                <AuxiliaryLabel<2, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Auxiliary4 => {
                <AuxiliaryLabel<4, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::Auxiliary8 => {
                <AuxiliaryLabel<8, D> as ArenaRecord<D>>::try_push_borrowed(self, label, local_id)
            }
            RecordKind::AuxiliaryFull => {
                <AuxiliaryLabel<LIMITED_WORDS, D> as ArenaRecord<D>>::try_push_borrowed(
                    self, label, local_id,
                )
            }
            RecordKind::Full => false,
        }
    }
}
