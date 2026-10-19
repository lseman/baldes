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

/// Branch-only labels retain no unused cut payload.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct BranchedLabel {
    base: OrdinaryLabel,
    branch_state: BranchState,
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

macro_rules! view_accessors {
    ($($name:ident: $ty:ty),* $(,)?) => {$(
        #[inline(always)]
        pub fn $name(self) -> $ty {
            match self { Self::Ordinary(l, _) => l.$name(), Self::Extended(l, _) => l.$name(), Self::Branched(l, _) => l.$name(), Self::Compact2(l, _) => l.$name(), Self::Compact4(l, _) => l.$name(), Self::Compact8(l, _) => l.$name(),  Self::Full(l) => l.$name() }
        }
    )*};
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

mod data;
mod inspection;
mod packing;
mod records;
mod view;

mod access;
mod allocation;
mod expansion;
