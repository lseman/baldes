use super::*;

impl<const D: usize> LabelStorage<D> {
    #[inline]
    pub fn dominates(
        g: &Graph<D>,
        a: &StoredLabel<D>,
        b: &StoredLabel<D>,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
    ) -> bool {
        a.presence_state == b.presence_state
            && a.branch_state == b.branch_state
            && Self::dominates_hot_slices(
                g,
                &HotLabel::of(a),
                &HotLabel::of(b),
                forward,
                test,
                hop_limit,
                cuts,
                || (&a.cut_words[..], &b.cut_words[..]),
            )
    }

    /// `dominates` on the packed fields; `words` loads the cut words only
    /// when full dominance reaches a nonempty cut list. Every scalar
    /// coordinate is held in the fixed-size hot cache.
    #[inline(always)]
    pub fn dominates_hot<'w>(
        g: &Graph<D>,
        a: &HotLabel<D>,
        b: &HotLabel<D>,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        words: impl FnOnce() -> (&'w [u32; LIMITED_WORDS], &'w [u32; LIMITED_WORDS]),
    ) -> bool {
        Self::dominates_hot_slices(g, a, b, forward, test, hop_limit, cuts, || {
            let (a, b) = words();
            (&a[..], &b[..])
        })
    }

    #[inline(always)]
    pub(crate) fn dominates_hot_slices<'w>(
        g: &Graph<D>,
        a: &HotLabel<D>,
        b: &HotLabel<D>,
        forward: bool,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        words: impl FnOnce() -> (&'w [u32], &'w [u32]),
    ) -> bool {
        if hop_limit && a.hops > b.hops {
            return false;
        }
        if !crate::labeling::ScalarResourceKernel::<D>::dominates(
            &g.resources,
            &a.resources,
            &b.resources,
            forward,
        ) {
            return false;
        }
        if matches!(test, InsertionTest::Full | InsertionTest::ResourcesAndNg) && a.ng & !b.ng != 0
        {
            return false;
        }
        match test {
            // Full dominance: the additional cost joins the gap, then
            // the vertex's packed cut fields.
            InsertionTest::Full => {
                let mut gap = (a.cost - b.cost) + (a.additional_cost - b.additional_cost);
                if gap >= EPSILON {
                    return false;
                }
                if cuts.is_empty() {
                    return true;
                }
                let (wa, wb) = words();
                packed_dominate(cuts, wa, wb, &mut gap)
            }
            InsertionTest::ResourcesAndCost => a.cost - b.cost < EPSILON,
            _ => true,
        }
    }

    /// `HotColumns::next_candidate` for this graph's resources; without a
    /// specialization every index is a candidate. Pass the candidate resources
    /// as scalars instead of reloading a freshly assembled `HotLabel`.
    #[inline]
    pub(super) fn next_candidate<const PREFIX: bool>(
        g: &Graph<D>,
        cols: &HotColumns<D>,
        from: usize,
        threshold: f64,
        candidate_r0: f64,
        candidate_r1: f64,
        candidate_ng: u16,
        forward: bool,
        ng: bool,
    ) -> usize {
        let disposable = g
            .resources
            .iter()
            .all(|k| matches!(k, ResourceKind::Disposable));
        macro_rules! scan {
            ($f:literal, $r:literal, $n:literal) => {
                cols.next_candidate::<PREFIX, $f, $r, $n>(
                    from,
                    threshold,
                    candidate_r0,
                    candidate_r1,
                    candidate_ng,
                )
            };
        }
        match (disposable, g.resources.len(), forward, ng) {
            (true, 1, true, true) => scan!(true, 1, true),
            (true, 1, true, false) => scan!(true, 1, false),
            (true, 1, false, true) => scan!(false, 1, true),
            (true, 1, false, false) => scan!(false, 1, false),
            (true, 2, true, true) => scan!(true, 2, true),
            (true, 2, true, false) => scan!(true, 2, false),
            (true, 2, false, true) => scan!(false, 2, true),
            (true, 2, false, false) => scan!(false, 2, false),
            _ => from,
        }
    }
}
