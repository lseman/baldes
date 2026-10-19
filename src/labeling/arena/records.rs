use super::*;

impl ArenaRecord for OrdinaryLabel {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        arena.ordinary_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Ordinary(v) = &mut arena.records {
            if let Some(compact) = OrdinaryLabel::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for ExtendedLabel {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        arena.extended_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Extended(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<LIMITED_WORDS>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for BranchedLabel {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        arena.branched_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Branched(v) = &mut arena.records {
            if let Some(compact) = BranchedLabel::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for ExtendedLabel<2> {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Compact2(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Compact2(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<2>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for ExtendedLabel<4> {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Compact4(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Compact4(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<4>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for ExtendedLabel<8> {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Compact8(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena, label: &mut StoredLabel) -> bool {
        if let Records::Compact8(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<8>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl ArenaRecord for StoredLabel {
    #[inline(always)]
    fn get(arena: &LabelArena, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Full(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.expanded
    }
    #[inline(always)]
    fn ancestry_id(&self, _index: usize) -> usize {
        self.id
    }
    #[inline(always)]
    fn push_fallback(arena: &mut LabelArena, label: StoredLabel) {
        // The public insertion adapter accepts every current arena format.
        arena.push(label);
    }
}
