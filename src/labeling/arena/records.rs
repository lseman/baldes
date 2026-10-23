use super::*;

impl<const D: usize> ArenaRecord<D> for OrdinaryLabel<D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        arena.ordinary_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Ordinary(v) = &mut arena.records {
            if let Some(compact) = OrdinaryLabel::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Ordinary(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for ExtendedLabel<LIMITED_WORDS, D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        arena.extended_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Extended(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<LIMITED_WORDS, D>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Extended(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for BranchedLabel<D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        arena.branched_record(id)
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Branched(v) = &mut arena.records {
            if let Some(compact) = BranchedLabel::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Branched(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for ExtendedLabel<2, D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
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
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Compact2(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<2, D>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Compact2(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for ExtendedLabel<4, D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
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
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Compact4(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<4, D>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Compact4(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for ExtendedLabel<8, D> {
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
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
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Compact8(v) = &mut arena.records {
            if let Some(compact) = ExtendedLabel::<8, D>::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Compact8(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for StoredLabel<D> {
    const OWNS_AUXILIARY: bool = true;
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
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
    fn push_fallback(arena: &mut LabelArena<D>, label: StoredLabel<D>) {
        // The public insertion adapter accepts every current arena format.
        arena.push(label);
    }
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        arena.try_push_borrowed_current(label, local_id)
    }
}

impl<const D: usize> ArenaRecord<D> for AuxiliaryLabel<2, D> {
    const OWNS_AUXILIARY: bool = true;
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Auxiliary2(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Auxiliary2(v) = &mut arena.records {
            if let Some(compact) = Self::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Auxiliary2(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for AuxiliaryLabel<4, D> {
    const OWNS_AUXILIARY: bool = true;
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Auxiliary4(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Auxiliary4(v) = &mut arena.records {
            if let Some(compact) = Self::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Auxiliary4(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for AuxiliaryLabel<8, D> {
    const OWNS_AUXILIARY: bool = true;
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::Auxiliary8(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::Auxiliary8(v) = &mut arena.records {
            if let Some(compact) = Self::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::Auxiliary8(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}

impl<const D: usize> ArenaRecord<D> for AuxiliaryLabel<LIMITED_WORDS, D> {
    const OWNS_AUXILIARY: bool = true;
    #[inline(always)]
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self> {
        match &arena.records {
            Records::AuxiliaryFull(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    fn expanded(&self) -> bool {
        self.base.expanded
    }
    #[inline(always)]
    fn try_push(arena: &mut LabelArena<D>, label: &mut StoredLabel<D>) -> bool {
        if let Records::AuxiliaryFull(v) = &mut arena.records {
            if let Some(compact) = Self::pack(label, v.len()) {
                v.push(compact);
                return true;
            }
        }
        false
    }
    #[inline(always)]
    fn try_push_borrowed(arena: &mut LabelArena<D>, label: &StoredLabel<D>, local_id: u64) -> bool {
        if let Records::AuxiliaryFull(v) = &mut arena.records {
            if let Some(compact) = Self::pack_retained(label, local_id) {
                v.push(compact);
                return true;
            }
        }
        false
    }
}
