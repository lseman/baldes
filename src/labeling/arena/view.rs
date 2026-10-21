use super::*;

impl<'a, const D: usize> From<&'a StoredLabel<D>> for LabelRef<'a, D> {
    #[inline(always)]
    fn from(l: &'a StoredLabel<D>) -> Self {
        Self::Full(l)
    }
}

impl<'a, const D: usize> LabelRef<'a, D> {
    view_accessors!(vertex: usize, cost: f64, original_cost: f64, additional_cost: f64, ng: u16, resources: &'a [f64],
        branch_state: &'a BranchState, presence_state: &'a crate::cuts::soft_conflicts::LabelState, hops: usize,
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
    pub fn to_owned(self) -> StoredLabel<D> {
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
