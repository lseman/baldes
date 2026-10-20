//! Typed label arenas selected from the active resource shape, with lossless
//! promotion to public wide records when compact metadata cannot represent a value.
use super::storage::StoredLabel;
use crate::cuts::state::LIMITED_WORDS;
use crate::graph::buckets::BucketId;
use crate::pricing::restrictions::BranchState;
use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

const NONE: u32 = u32::MAX;
static EMPTY_WORDS: [u32; LIMITED_WORDS] = [0; LIMITED_WORDS];
static EMPTY_BRANCH: BranchState = None;

/// Only ordinary-label state. IDs remain stable across arena reallocations;
/// identifiers that cannot be represented exactly select full records.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrdinaryLabel<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    cost: f64,
    original_cost: f64,
    resources: [f64; D],
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

/// Compact metadata plus every extended resource, without narrowing floats.
// Keep shared metadata at offset zero for ordinary and extended accessors.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct ExtendedLabel<
    const WORDS: usize = LIMITED_WORDS,
    const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS },
> {
    base: OrdinaryLabel<D>,
    cut_words: [u32; WORDS],
    additional_cost: f64,
    half_potential: f64,
    branch_state: BranchState,
}

/// Branch-only labels retain no unused cut payload.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct BranchedLabel<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    base: OrdinaryLabel<D>,
    branch_state: BranchState,
}

/// Accessors specialize to direct field reads for ordinary and full records.
/// The hot kernels use this trait statically, without virtual calls.
pub(crate) trait LabelData {
    fn vertex(&self) -> usize;
    fn cost(&self) -> f64;
    fn original_cost(&self) -> f64;
    fn resources(&self) -> &[f64];
    fn ng(&self) -> u16;
    fn branch_state(&self) -> &BranchState;
    fn hops(&self) -> usize;
    fn cut_words(&self) -> &[u32];
    fn additional_cost(&self) -> f64;
    fn key(&self) -> f64;
}

/// Select the typed sweep once; promotion can only select wide records.
pub(crate) trait ArenaRecord<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }>:
    LabelData + 'static
{
    fn get(arena: &LabelArena<D>, id: usize) -> Option<&Self>;
    fn expanded(&self) -> bool;
    fn try_push(_arena: &mut LabelArena<D>, _label: &mut StoredLabel<D>) -> bool {
        false
    }
    #[inline(always)]
    fn push_fallback(arena: &mut LabelArena<D>, label: StoredLabel<D>) {
        arena.push_wide(label);
    }
    fn ancestry_id(&self, index: usize) -> usize {
        index
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
pub(crate) enum LabelRef<'a, const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    Ordinary(&'a OrdinaryLabel<D>, usize),
    Extended(&'a ExtendedLabel<LIMITED_WORDS, D>, usize),
    Branched(&'a BranchedLabel<D>, usize),
    Compact2(&'a ExtendedLabel<2, D>, usize),
    Compact4(&'a ExtendedLabel<4, D>, usize),
    Compact8(&'a ExtendedLabel<8, D>, usize),
    Full(&'a StoredLabel<D>),
}

macro_rules! view_accessors {
    ($($name:ident: $ty:ty),* $(,)?) => {$(
        #[inline(always)]
        pub fn $name(self) -> $ty {
            match self { Self::Ordinary(l, _) => l.$name(), Self::Extended(l, _) => l.$name(), Self::Branched(l, _) => l.$name(), Self::Compact2(l, _) => l.$name(), Self::Compact4(l, _) => l.$name(), Self::Compact8(l, _) => l.$name(),  Self::Full(l) => l.$name() }
        }
    )*};
}

#[derive(Clone, Debug)]
enum Records<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    Ordinary(Vec<OrdinaryLabel<D>>),
    Extended(Vec<ExtendedLabel<LIMITED_WORDS, D>>),
    Branched(Vec<BranchedLabel<D>>),
    Compact2(Vec<ExtendedLabel<2, D>>),
    Compact4(Vec<ExtendedLabel<4, D>>),
    Compact8(Vec<ExtendedLabel<8, D>>),
    Full(Vec<StoredLabel<D>>),
}

/// Arena with 64-byte ordinary, 72-byte branch-only, and cut records sized to
/// the prepared state (96, 104, 120, or 168 bytes on 64-bit targets). Existing
/// vector inspection remains available through `Deref`: a read materializes
/// full records lazily, and mutable vector access promotes the arena to full
/// records. Internal labeling uses borrowed views and never needs that copy.
#[derive(Debug)]
pub struct LabelArena<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    records: Records<D>,
    inspected: OnceLock<Vec<StoredLabel<D>>>,
    word_limit: usize,
}
impl<const D: usize> Clone for LabelArena<D> {
    fn clone(&self) -> Self {
        Self {
            records: self.records.clone(),
            word_limit: self.word_limit,
            inspected: OnceLock::new(),
        }
    }
}
impl<const D: usize> Default for LabelArena<D> {
    fn default() -> Self {
        Self {
            records: Records::Full(vec![]),
            word_limit: LIMITED_WORDS,
            inspected: OnceLock::new(),
        }
    }
}

mod data;
mod inspection;
mod packing;
mod records;
mod view;

mod access;
mod allocation;
mod expansion;
