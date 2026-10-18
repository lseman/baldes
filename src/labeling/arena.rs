//! Typed label arenas selected from the active resource shape, with lossless
//! promotion to public wide records when compact metadata cannot represent a value.
use super::storage::StoredLabel;
use crate::cuts::state::LIMITED_WORDS;
use crate::graph::buckets::BucketId;
use crate::pricing::branching::BranchState;
use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

const NONE: u32 = u32::MAX;
static EMPTY_WORDS: [u32; LIMITED_WORDS] = [0; LIMITED_WORDS];
static EMPTY_BRANCH: BranchState = None;

/// Only ordinary-label state. IDs remain stable across arena reallocations;
/// identifiers that cannot be represented exactly select full records.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrdinaryLabel {
    cost: f64,
    original_cost: f64,
    resources: [f64; 2],
    parent: u32,
    arc: u32,
    bucket_vertex: u32,
    bucket: u32,
    vertex: u32,
    hops: u32,
    local_id: u32,
    ng: u16,
    expanded: bool,
}
impl OrdinaryLabel {
    #[inline(always)]
    fn pack(label: &StoredLabel, id: usize) -> Option<Self> {
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
    fn pack_metadata(label: &StoredLabel, id: usize) -> Option<Self> {
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
            resources: label.resources,
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
    fn view(&self, id: usize) -> LabelRef<'_> {
        LabelRef::Ordinary(self, id)
    }
    fn unpack(self, id: usize) -> StoredLabel {
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
            resources: self.resources,
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

/// Compact metadata plus every extended resource, without narrowing floats.
// Keep shared metadata at offset zero for ordinary and extended accessors.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct ExtendedLabel<const WORDS: usize = LIMITED_WORDS> {
    base: OrdinaryLabel,
    cut_words: [u32; WORDS],
    additional_cost: f64,
    half_potential: f64,
    branch_state: BranchState,
}
impl<const WORDS: usize> ExtendedLabel<WORDS> {
    #[inline(always)]
    fn pack(label: &mut StoredLabel, id: usize) -> Option<Self> {
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
    fn unpack(&self, id: usize) -> StoredLabel {
        let mut label = self.base.unpack(id);
        label.cut_words[..WORDS].copy_from_slice(&self.cut_words);
        label.additional_cost = self.additional_cost;
        label.half_potential = self.half_potential;
        label.branch_state = self.branch_state.clone();
        label
    }
}
impl<const WORDS: usize> LabelData for ExtendedLabel<WORDS> {
    #[inline(always)]
    fn vertex(&self) -> usize {
        self.base.vertex()
    }
    #[inline(always)]
    fn cost(&self) -> f64 {
        self.base.cost()
    }
    #[inline(always)]
    fn original_cost(&self) -> f64 {
        self.base.original_cost()
    }
    #[inline(always)]
    fn resources(&self) -> &[f64; 2] {
        self.base.resources()
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        self.base.ng()
    }
    #[inline(always)]
    fn hops(&self) -> usize {
        self.base.hops()
    }
    #[inline(always)]
    fn branch_state(&self) -> &BranchState {
        &self.branch_state
    }
    #[inline(always)]
    fn cut_words(&self) -> &[u32] {
        &self.cut_words
    }
    #[inline(always)]
    fn additional_cost(&self) -> f64 {
        self.additional_cost
    }
    #[inline(always)]
    fn key(&self) -> f64 {
        (self.half_potential + self.additional_cost) + (0.0 + self.base.cost)
    }
}

/// Branch-only labels retain no unused cut payload.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct BranchedLabel {
    base: OrdinaryLabel,
    branch_state: BranchState,
}
impl BranchedLabel {
    #[inline(always)]
    fn pack(label: &mut StoredLabel, id: usize) -> Option<Self> {
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
    fn unpack(&self, id: usize) -> StoredLabel {
        let mut label = self.base.unpack(id);
        label.branch_state = self.branch_state.clone();
        label
    }
}
impl LabelData for BranchedLabel {
    #[inline(always)]
    fn vertex(&self) -> usize {
        self.base.vertex()
    }
    #[inline(always)]
    fn cost(&self) -> f64 {
        self.base.cost()
    }
    #[inline(always)]
    fn original_cost(&self) -> f64 {
        self.base.original_cost()
    }
    #[inline(always)]
    fn resources(&self) -> &[f64; 2] {
        self.base.resources()
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        self.base.ng()
    }
    #[inline(always)]
    fn hops(&self) -> usize {
        self.base.hops()
    }
    #[inline(always)]
    fn branch_state(&self) -> &BranchState {
        &self.branch_state
    }
    #[inline(always)]
    fn cut_words(&self) -> &[u32] {
        &EMPTY_WORDS
    }
    #[inline(always)]
    fn additional_cost(&self) -> f64 {
        0.
    }
    #[inline(always)]
    fn key(&self) -> f64 {
        self.base.key()
    }
}

/// Accessors specialize to direct field reads for ordinary and full records.
/// The hot kernels use this trait statically, without virtual calls.
pub(crate) trait LabelData {
    fn vertex(&self) -> usize;
    fn cost(&self) -> f64;
    fn original_cost(&self) -> f64;
    fn resources(&self) -> &[f64; 2];
    fn ng(&self) -> u16;
    fn branch_state(&self) -> &BranchState;
    fn hops(&self) -> usize;
    fn cut_words(&self) -> &[u32];
    fn additional_cost(&self) -> f64;
    fn key(&self) -> f64;
}
impl LabelData for OrdinaryLabel {
    #[inline(always)]
    fn vertex(&self) -> usize {
        self.vertex as usize
    }
    #[inline(always)]
    fn cost(&self) -> f64 {
        self.cost
    }
    #[inline(always)]
    fn original_cost(&self) -> f64 {
        self.original_cost
    }
    #[inline(always)]
    fn resources(&self) -> &[f64; 2] {
        &self.resources
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        self.ng
    }
    #[inline(always)]
    fn branch_state(&self) -> &BranchState {
        &EMPTY_BRANCH
    }
    #[inline(always)]
    fn hops(&self) -> usize {
        self.hops as usize
    }
    #[inline(always)]
    fn cut_words(&self) -> &[u32] {
        &EMPTY_WORDS
    }
    #[inline(always)]
    fn additional_cost(&self) -> f64 {
        0.
    }
    #[inline(always)]
    fn key(&self) -> f64 {
        (0.0 + 0.0) + (0.0 + self.cost)
    }
}
impl LabelData for StoredLabel {
    #[inline(always)]
    fn vertex(&self) -> usize {
        self.vertex
    }
    #[inline(always)]
    fn cost(&self) -> f64 {
        self.cost
    }
    #[inline(always)]
    fn original_cost(&self) -> f64 {
        self.original_cost
    }
    #[inline(always)]
    fn resources(&self) -> &[f64; 2] {
        &self.resources
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        self.ng
    }
    #[inline(always)]
    fn branch_state(&self) -> &BranchState {
        &self.branch_state
    }
    #[inline(always)]
    fn hops(&self) -> usize {
        self.hops
    }
    #[inline(always)]
    fn cut_words(&self) -> &[u32] {
        &self.cut_words
    }
    #[inline(always)]
    fn additional_cost(&self) -> f64 {
        self.additional_cost
    }
    #[inline(always)]
    fn key(&self) -> f64 {
        StoredLabel::key(self)
    }
}

/// Select the typed sweep once; promotion can only select wide records.
pub(crate) trait ArenaRecord: LabelData + 'static {
    fn get(arena: &LabelArena, id: usize) -> Option<&Self>;
    fn expanded(&self) -> bool;
    fn try_push(_arena: &mut LabelArena, _label: &mut StoredLabel) -> bool {
        false
    }
    #[inline(always)]
    fn push_fallback(arena: &mut LabelArena, label: StoredLabel) {
        arena.push_wide(label);
    }
    fn ancestry_id(&self, index: usize) -> usize {
        index
    }
}
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordKind {
    Ordinary,
    Extended,
    Branched,
    Compact2,
    Compact4,
    Compact8,
    Full,
}

/// A borrowed record; extended state is never copied merely to inspect it.
#[derive(Clone, Copy)]
pub(crate) enum LabelRef<'a> {
    Ordinary(&'a OrdinaryLabel, usize),
    Extended(&'a ExtendedLabel, usize),
    Branched(&'a BranchedLabel, usize),
    Compact2(&'a ExtendedLabel<2>, usize),
    Compact4(&'a ExtendedLabel<4>, usize),
    Compact8(&'a ExtendedLabel<8>, usize),
    Full(&'a StoredLabel),
}
impl<'a> From<&'a StoredLabel> for LabelRef<'a> {
    #[inline(always)]
    fn from(l: &'a StoredLabel) -> Self {
        Self::Full(l)
    }
}
macro_rules! view_accessors {
    ($($name:ident: $ty:ty),* $(,)?) => {$(
        #[inline(always)]
        pub fn $name(self) -> $ty {
            match self { Self::Ordinary(l, _) => l.$name(), Self::Extended(l, _) => l.$name(), Self::Branched(l, _) => l.$name(), Self::Compact2(l, _) => l.$name(), Self::Compact4(l, _) => l.$name(), Self::Compact8(l, _) => l.$name(),  Self::Full(l) => l.$name() }
        }
    )*};
}
impl<'a> LabelRef<'a> {
    view_accessors!(vertex: usize, cost: f64, original_cost: f64, additional_cost: f64, ng: u16, resources: &'a [f64; 2],
        branch_state: &'a BranchState, hops: usize,
        cut_words: &'a [u32], key: f64);
    #[inline(always)]
    pub fn expanded(self) -> bool {
        match self {
            Self::Ordinary(l, _) => l.expanded,
            Self::Extended(l, _) => l.base.expanded,
            Self::Branched(l, _) => l.base.expanded,
            Self::Compact2(l, _) => l.base.expanded,
            Self::Compact4(l, _) => l.base.expanded,
            Self::Compact8(l, _) => l.base.expanded,
            Self::Full(l) => l.expanded,
        }
    }
    #[inline(always)]
    pub fn bucket(self) -> BucketId {
        match self {
            Self::Ordinary(l, _) => BucketId {
                vertex: l.bucket_vertex as usize,
                bucket: l.bucket as usize,
            },
            Self::Extended(l, _) => BucketId {
                vertex: l.base.bucket_vertex as usize,
                bucket: l.base.bucket as usize,
            },
            Self::Branched(l, _) => BucketId {
                vertex: l.base.bucket_vertex as usize,
                bucket: l.base.bucket as usize,
            },
            Self::Compact2(l, _) => BucketId {
                vertex: l.base.bucket_vertex as usize,
                bucket: l.base.bucket as usize,
            },
            Self::Compact4(l, _) => BucketId {
                vertex: l.base.bucket_vertex as usize,
                bucket: l.base.bucket as usize,
            },
            Self::Compact8(l, _) => BucketId {
                vertex: l.base.bucket_vertex as usize,
                bucket: l.base.bucket as usize,
            },
            Self::Full(l) => l.bucket,
        }
    }
    #[inline(always)]
    pub fn parent(self) -> Option<usize> {
        match self {
            Self::Ordinary(l, _) => (l.parent != NONE).then_some(l.parent as usize),
            Self::Extended(l, _) => (l.base.parent != NONE).then_some(l.base.parent as usize),
            Self::Branched(l, _) => (l.base.parent != NONE).then_some(l.base.parent as usize),
            Self::Compact2(l, _) => (l.base.parent != NONE).then_some(l.base.parent as usize),
            Self::Compact4(l, _) => (l.base.parent != NONE).then_some(l.base.parent as usize),
            Self::Compact8(l, _) => (l.base.parent != NONE).then_some(l.base.parent as usize),
            Self::Full(l) => l.parent,
        }
    }
    #[inline(always)]
    pub fn arc(self) -> Option<usize> {
        match self {
            Self::Ordinary(l, _) => (l.arc != NONE).then_some(l.arc as usize),
            Self::Extended(l, _) => (l.base.arc != NONE).then_some(l.base.arc as usize),
            Self::Branched(l, _) => (l.base.arc != NONE).then_some(l.base.arc as usize),
            Self::Compact2(l, _) => (l.base.arc != NONE).then_some(l.base.arc as usize),
            Self::Compact4(l, _) => (l.base.arc != NONE).then_some(l.base.arc as usize),
            Self::Compact8(l, _) => (l.base.arc != NONE).then_some(l.base.arc as usize),
            Self::Full(l) => l.arc,
        }
    }
    pub fn to_owned(self) -> StoredLabel {
        match self {
            Self::Ordinary(l, id) => l.unpack(id),
            Self::Extended(l, id) => l.unpack(id),
            Self::Branched(l, id) => l.unpack(id),
            Self::Compact2(l, id) => l.unpack(id),
            Self::Compact4(l, id) => l.unpack(id),
            Self::Compact8(l, id) => l.unpack(id),
            Self::Full(l) => l.clone(),
        }
    }
}

impl LabelData for LabelRef<'_> {
    #[inline(always)]
    fn vertex(&self) -> usize {
        (*self).vertex()
    }
    #[inline(always)]
    fn cost(&self) -> f64 {
        (*self).cost()
    }
    #[inline(always)]
    fn original_cost(&self) -> f64 {
        (*self).original_cost()
    }
    #[inline(always)]
    fn resources(&self) -> &[f64; 2] {
        (*self).resources()
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        (*self).ng()
    }
    #[inline(always)]
    fn branch_state(&self) -> &BranchState {
        (*self).branch_state()
    }
    #[inline(always)]
    fn hops(&self) -> usize {
        (*self).hops()
    }
    #[inline(always)]
    fn cut_words(&self) -> &[u32] {
        (*self).cut_words()
    }
    #[inline(always)]
    fn additional_cost(&self) -> f64 {
        (*self).additional_cost()
    }
    #[inline(always)]
    fn key(&self) -> f64 {
        (*self).key()
    }
}

#[derive(Clone, Debug)]
enum Records {
    Ordinary(Vec<OrdinaryLabel>),
    Extended(Vec<ExtendedLabel>),
    Branched(Vec<BranchedLabel>),
    Compact2(Vec<ExtendedLabel<2>>),
    Compact4(Vec<ExtendedLabel<4>>),
    Compact8(Vec<ExtendedLabel<8>>),
    Full(Vec<StoredLabel>),
}

/// Arena with 64-byte ordinary, 72-byte branch-only, and cut records sized to
/// the prepared state (96, 104, 120, or 168 bytes on 64-bit targets). Existing
/// vector inspection remains available through `Deref`: a read materializes
/// full records lazily, and mutable vector access promotes the arena to full
/// records. Internal labeling uses borrowed views and never needs that copy.
#[derive(Debug)]
pub struct LabelArena {
    records: Records,
    inspected: OnceLock<Vec<StoredLabel>>,
    word_limit: usize,
}
impl Clone for LabelArena {
    fn clone(&self) -> Self {
        Self {
            records: self.records.clone(),
            word_limit: self.word_limit,
            inspected: OnceLock::new(),
        }
    }
}
impl Default for LabelArena {
    fn default() -> Self {
        Self {
            records: Records::Full(vec![]),
            word_limit: LIMITED_WORDS,
            inspected: OnceLock::new(),
        }
    }
}
impl LabelArena {
    pub fn len(&self) -> usize {
        match &self.records {
            Records::Ordinary(v) => v.len(),
            Records::Extended(v) => v.len(),
            Records::Branched(v) => v.len(),
            Records::Compact2(v) => v.len(),
            Records::Compact4(v) => v.len(),
            Records::Compact8(v) => v.len(),
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
            Records::Full(v) => v.capacity(),
        }
    }
    /// Heap capacity of retained records, including a lazily inspected copy.
    pub fn allocated_bytes(&self) -> usize {
        let records = match &self.records {
            Records::Ordinary(v) => v.capacity() * std::mem::size_of::<OrdinaryLabel>(),
            Records::Extended(v) => v.capacity() * std::mem::size_of::<ExtendedLabel>(),
            Records::Branched(v) => v.capacity() * std::mem::size_of::<BranchedLabel>(),
            Records::Compact2(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<2>>(),
            Records::Compact4(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<4>>(),
            Records::Compact8(v) => v.capacity() * std::mem::size_of::<ExtendedLabel<8>>(),
            Records::Full(v) => v.capacity() * std::mem::size_of::<StoredLabel>(),
        };
        records
            + self
                .inspected
                .get()
                .map_or(0, |v| v.capacity() * std::mem::size_of::<StoredLabel>())
    }
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
    /// Select representation after clearing dual-dependent labels. Repeated
    /// pricing calls retain and reuse their compact allocation.
    #[cfg(test)]
    pub(crate) fn set_ordinary(&mut self, ordinary: bool) {
        self.set_mode(ordinary, false);
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
    pub fn push(&mut self, mut label: StoredLabel) {
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
                if let Some(compact) = ExtendedLabel::<LIMITED_WORDS>::pack(&mut label, v.len()) {
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
                if let Some(compact) = ExtendedLabel::<2>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Compact4(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<4>::pack(&mut label, v.len()) {
                    v.push(compact);
                    return;
                }
            }
            Records::Compact8(v) => {
                self.inspected.take();
                if let Some(compact) = ExtendedLabel::<8>::pack(&mut label, v.len()) {
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
    pub(crate) fn kind(&self) -> RecordKind {
        match self.records {
            Records::Ordinary(_) => RecordKind::Ordinary,
            Records::Extended(_) => RecordKind::Extended,
            Records::Branched(_) => RecordKind::Branched,
            Records::Compact2(_) => RecordKind::Compact2,
            Records::Compact4(_) => RecordKind::Compact4,
            Records::Compact8(_) => RecordKind::Compact8,
            Records::Full(_) => RecordKind::Full,
        }
    }
    #[inline(always)]
    pub(crate) fn wide_record(&self, id: usize) -> &StoredLabel {
        match &self.records {
            Records::Full(v) => &v[id],
            _ => unreachable!("unexpected label representation during a typed sweep"),
        }
    }
    #[inline(always)]
    pub(crate) fn push_for<R: ArenaRecord>(&mut self, mut label: StoredLabel) {
        self.inspected.take();
        if !R::try_push(self, &mut label) {
            R::push_fallback(self, label);
        }
    }
    #[inline(never)]
    fn push_wide(&mut self, label: StoredLabel) {
        self.full_mut().push(label);
    }
    #[inline(always)]
    pub(crate) fn words_for<R: ArenaRecord>(&self, id: usize) -> &[u32] {
        if let Some(label) = R::get(self, id) {
            label.cut_words()
        } else {
            self.view(id).cut_words()
        }
    }
    #[inline(always)]
    pub(crate) fn branch_for<R: ArenaRecord>(&self, id: usize) -> &BranchState {
        if let Some(label) = R::get(self, id) {
            label.branch_state()
        } else {
            self.view(id).branch_state()
        }
    }
    /// Borrow ordinary metadata directly in the sweep. Promotion remains safe:
    /// once an identifier needs wide storage, callers use the general view.
    #[inline(always)]
    pub(crate) fn ordinary_record(&self, id: usize) -> Option<&OrdinaryLabel> {
        match &self.records {
            Records::Ordinary(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    pub(crate) fn extended_record(&self, id: usize) -> Option<&ExtendedLabel> {
        match &self.records {
            Records::Extended(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    pub(crate) fn branched_record(&self, id: usize) -> Option<&BranchedLabel> {
        match &self.records {
            Records::Branched(v) => Some(&v[id]),
            _ => None,
        }
    }
    #[inline(always)]
    pub(crate) fn view(&self, id: usize) -> LabelRef<'_> {
        match &self.records {
            Records::Ordinary(v) => v[id].view(id),
            Records::Extended(v) => LabelRef::Extended(&v[id], id),
            Records::Branched(v) => LabelRef::Branched(&v[id], id),
            Records::Compact2(v) => LabelRef::Compact2(&v[id], id),
            Records::Compact4(v) => LabelRef::Compact4(&v[id], id),
            Records::Compact8(v) => LabelRef::Compact8(&v[id], id),
            Records::Full(v) => (&v[id]).into(),
        }
    }
    pub(crate) fn expand(&mut self, ids: &[usize]) -> usize {
        let mut count = 0;
        match &mut self.records {
            Records::Ordinary(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].expanded) as usize;
                    v[id].expanded = true;
                }
            }
            Records::Extended(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Branched(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact2(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact4(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Compact8(v) => {
                self.inspected.take();
                for &id in ids {
                    count += (!v[id].base.expanded) as usize;
                    v[id].base.expanded = true;
                }
            }
            Records::Full(v) => {
                for &id in ids {
                    count += (!v[id].expanded) as usize;
                    v[id].expanded = true;
                }
            }
        }
        count
    }
    pub(crate) fn unexpand_where(&mut self, mut predicate: impl FnMut([f64; 2]) -> bool) {
        self.inspected.take();
        match &mut self.records {
            Records::Ordinary(v) => {
                for label in v {
                    if predicate(label.resources) {
                        label.expanded = false;
                    }
                }
            }
            Records::Extended(v) => {
                for label in v {
                    if predicate(label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Branched(v) => {
                for label in v {
                    if predicate(label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact2(v) => {
                for label in v {
                    if predicate(label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact4(v) => {
                for label in v {
                    if predicate(label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Compact8(v) => {
                for label in v {
                    if predicate(label.base.resources) {
                        label.base.expanded = false;
                    }
                }
            }
            Records::Full(v) => {
                for label in v {
                    if predicate(label.resources) {
                        label.expanded = false;
                    }
                }
            }
        }
    }
    fn full_mut(&mut self) -> &mut Vec<StoredLabel> {
        if !matches!(self.records, Records::Full(_)) {
            let full = self
                .inspected
                .take()
                .unwrap_or_else(|| (0..self.len()).map(|id| self.view(id).to_owned()).collect());
            self.records = Records::Full(full);
        }
        match &mut self.records {
            Records::Full(v) => v,
            Records::Ordinary(_)
            | Records::Extended(_)
            | Records::Branched(_)
            | Records::Compact2(_)
            | Records::Compact4(_)
            | Records::Compact8(_) => unreachable!(),
        }
    }
}
impl Deref for LabelArena {
    type Target = Vec<StoredLabel>;
    fn deref(&self) -> &Self::Target {
        match &self.records {
            Records::Full(v) => v,
            Records::Compact2(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Compact4(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Compact8(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Branched(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Extended(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Ordinary(v) => self.inspected.get_or_init(|| {
                v.iter()
                    .enumerate()
                    .map(|(id, l)| l.view(id).to_owned())
                    .collect()
            }),
        }
    }
}
impl DerefMut for LabelArena {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.full_mut()
    }
}
impl<'a> IntoIterator for &'a LabelArena {
    type Item = &'a StoredLabel;
    type IntoIter = std::slice::Iter<'a, StoredLabel>;
    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}
impl<'a> IntoIterator for &'a mut LabelArena {
    type Item = &'a mut StoredLabel;
    type IntoIter = std::slice::IterMut<'a, StoredLabel>;
    fn into_iter(self) -> Self::IntoIter {
        self.full_mut().iter_mut()
    }
}
impl IntoIterator for LabelArena {
    type Item = StoredLabel;
    type IntoIter = std::vec::IntoIter<StoredLabel>;
    fn into_iter(mut self) -> Self::IntoIter {
        std::mem::take(self.full_mut()).into_iter()
    }
}

#[cfg(test)]
mod tests;
