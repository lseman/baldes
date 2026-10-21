use super::*;

impl<const WORDS: usize, const D: usize> LabelData for ExtendedLabel<WORDS, D> {
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
    fn resources(&self) -> &[f64] {
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

impl<const D: usize> LabelData for BranchedLabel<D> {
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
    fn resources(&self) -> &[f64] {
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

impl<const D: usize> LabelData for OrdinaryLabel<D> {
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
    fn resources(&self) -> &[f64] {
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

impl<const D: usize> LabelData for StoredLabel<D> {
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
    fn resources(&self) -> &[f64] {
        &self.resources
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        self.ng
    }
    #[inline(always)]
    fn presence_state(&self) -> &crate::cuts::soft_conflicts::LabelState {
        &self.presence_state
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

impl<const D: usize> LabelData for LabelRef<'_, D> {
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
    fn resources(&self) -> &[f64] {
        (*self).resources()
    }
    #[inline(always)]
    fn ng(&self) -> u16 {
        (*self).ng()
    }
    #[inline(always)]
    fn presence_state(&self) -> &crate::cuts::soft_conflicts::LabelState {
        (*self).presence_state()
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
