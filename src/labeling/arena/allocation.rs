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
            Records::Full(v) => v.clear(),
        }
    }

    pub(crate) fn set_word_limit(&mut self, words: usize) {
        assert!(words <= LIMITED_WORDS);
        self.word_limit = words;
    }

    pub(crate) fn set_mode(&mut self, ordinary: bool, branch_only: bool) {
        assert!(
            self.is_empty(),
            "label representation changes require an empty arena"
        );
        assert!(!(ordinary && branch_only));
        self.inspected.take();
        let target = if ordinary {
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
