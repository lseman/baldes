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
        a.vertex == b.vertex
            && a.presence_state == b.presence_state
            && a.branch_state == b.branch_state
            && Self::dominates_coupled_slices(
                g,
                &HotLabel::of(a),
                &HotLabel::of(b),
                forward,
                a.vertex,
                test,
                hop_limit,
                cuts,
                a.coupled_state.as_deref(),
                b.coupled_state.as_deref(),
                || (&a.cut_words[..], &b.cut_words[..]),
            )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)] // Scalar hot-scan parameters plus the two coupled states.
    pub(crate) fn dominates_coupled_slices<'w>(
        g: &Graph<D>,
        a: &HotLabel<D>,
        b: &HotLabel<D>,
        forward: bool,
        vertex: usize,
        test: InsertionTest,
        hop_limit: bool,
        cuts: &[DominanceDescriptor],
        a_state: Option<&crate::graph::coupled::CoupledLabelState>,
        b_state: Option<&crate::graph::coupled::CoupledLabelState>,
        words: impl FnOnce() -> (&'w [u32], &'w [u32]),
    ) -> bool {
        let Some(penalty) = g.coupled_dominance_penalty(a_state, b_state, forward, vertex) else {
            return false;
        };
        let retained_fronts = g
            .coupled_kernel()
            .map_or(false, |k| k.retains_cost_components());
        let (cached_words, pending_words) = if retained_fronts {
            // Rounded common future costs can shrink the current mixed-key
            // gap. Do not spend that gap on different future cut behavior or
            // on an oppositely ordered additional-cost component.
            if a.additional_cost > b.additional_cost {
                return false;
            }
            let pair = words();
            if pair.0 != pair.1 {
                return false;
            }
            (Some(pair), None)
        } else {
            (None, Some(words))
        };
        // Read the cold words at most once, retaining lazy access for the
        // ordinary scalar-only hot path.
        let words = || cached_words.unwrap_or_else(|| pending_words.expect("uncached cut words")());
        if penalty == 0. {
            return Self::dominates_hot_slices(g, a, b, forward, test, hop_limit, cuts, words);
        }
        let Some(cost) = crate::core::rounding::add_up(a.cost, penalty) else {
            return false;
        };
        let adjusted = HotLabel { cost, ..*a };
        Self::dominates_hot_slices(g, &adjusted, b, forward, test, hop_limit, cuts, words)
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
        // Hot records alone cannot establish a coupled state relation.
        if g.has_coupled_resources() {
            return false;
        }
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

#[cfg(test)]
mod retained_front_cut_tests {
    use super::*;
    use crate::graph::coupled::{CoupledResource, PiecewiseLinearFunction};
    use crate::Window;

    #[test]
    fn front_cost_margin_cannot_pay_cut_state_tradeoffs_and_words_are_read_once() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head, cost) in [(0, 1, -2.), (0, 1, 0.), (1, 2, 0.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 1.).unwrap(),
                increments: vec![0.; 3],
                functions: vec![PiecewiseLinearFunction::new(vec![[0., 0.], [1., 0.]]).unwrap(); 3],
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (mut left, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (mut right, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        let (left_cost, _) = left.account_extension(&root, -2., -2.).unwrap();
        let (right_cost, _) = right.account_extension(&root, 0., 0.).unwrap();
        let base = HotLabel {
            key: 0.,
            cost: right_cost,
            additional_cost: 0.,
            resources: [0.; 2],
            hops: 1,
            ng: 0,
        };
        let mut a = HotLabel {
            cost: left_cost,
            additional_cost: 2.,
            ..base
        };
        let words = [0u32; 1];
        assert!(!LabelStorage::dominates_coupled_slices(
            &graph,
            &a,
            &base,
            true,
            1,
            InsertionTest::Full,
            false,
            &[],
            Some(&left),
            Some(&right),
            || (&words[..], &words[..]),
        ));
        a.additional_cost = 0.;
        let different = [1u32; 1];
        assert!(!LabelStorage::dominates_coupled_slices(
            &graph,
            &a,
            &base,
            true,
            1,
            InsertionTest::Full,
            false,
            &[],
            Some(&left),
            Some(&right),
            || (&different[..], &words[..]),
        ));
        let reads = std::cell::Cell::new(0);
        assert!(LabelStorage::dominates_coupled_slices(
            &graph,
            &a,
            &base,
            true,
            1,
            InsertionTest::Full,
            false,
            &[],
            Some(&left),
            Some(&right),
            || {
                reads.set(reads.get() + 1);
                (&words[..], &words[..])
            },
        ));
        assert_eq!(reads.get(), 1);
    }
}
