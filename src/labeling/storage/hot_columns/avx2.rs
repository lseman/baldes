use super::*;

impl<const D: usize> HotColumns<D> {
    /// Four-label AVX2 kernel; comparisons retain their original EPSILON order.
    ///
    /// # Safety
    /// The caller must establish AVX2 support on the executing CPU. Column
    /// lengths are validated here before any pointer loads.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn next_candidate_avx2<
        const PREFIX: bool,
        const FORWARD: bool,
        const RESOURCES: usize,
        const NG: bool,
    >(
        &self,
        from: usize,
        threshold: f64,
        candidate_r0: f64,
        candidate_r1: f64,
        candidate_ng: NgMask,
    ) -> usize {
        use std::arch::x86_64::*;
        let n = self.key.len();
        // Validate public columns before any unaligned vector loads.
        assert!(
            D >= 2
                && self.resources[0].len() >= n
                && self.resources[1].len() >= n
                && self.ng.len() >= n
        );
        let mut i = from;
        // SAFETY: caller establishes AVX2; each loop loads four elements within
        // the checked column lengths. loadu requires no extra alignment.
        unsafe {
            let eps = _mm256_set1_pd(EPSILON);
            while i < n && n - i >= 4 {
                let mut pass = 15;
                for (r, candidate) in [candidate_r0, candidate_r1]
                    .into_iter()
                    .enumerate()
                    .take(RESOURCES)
                {
                    let stored = _mm256_loadu_pd(self.resources[r].as_ptr().add(i));
                    let c = _mm256_set1_pd(candidate);
                    let mask = if PREFIX {
                        if FORWARD {
                            _mm256_cmp_pd::<_CMP_LE_OQ>(stored, _mm256_add_pd(c, eps))
                        } else {
                            _mm256_cmp_pd::<_CMP_LE_OQ>(_mm256_sub_pd(c, eps), stored)
                        }
                    } else if FORWARD {
                        _mm256_cmp_pd::<_CMP_LE_OQ>(c, _mm256_add_pd(stored, eps))
                    } else {
                        _mm256_cmp_pd::<_CMP_LE_OQ>(_mm256_sub_pd(stored, eps), c)
                    };
                    pass &= _mm256_movemask_pd(mask);
                }
                if NG {
                    // Widen four NG masks to four u64 lanes.
                    let ng = self.ng.as_ptr().add(i).cast();
                    #[cfg(not(any(feature = "ng-memory-31", feature = "ng-memory-63")))]
                    let stored = _mm256_cvtepu16_epi64(_mm_loadl_epi64(ng));
                    #[cfg(all(feature = "ng-memory-31", not(feature = "ng-memory-63")))]
                    let stored = _mm256_cvtepu32_epi64(_mm_loadu_si128(ng));
                    #[cfg(feature = "ng-memory-63")]
                    let stored = _mm256_loadu_si256(ng);
                    let c = _mm256_set1_epi64x(candidate_ng as i64);
                    let excluded = if PREFIX {
                        _mm256_andnot_si256(c, stored)
                    } else {
                        _mm256_andnot_si256(stored, c)
                    };
                    pass &= _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                        excluded,
                        _mm256_setzero_si256(),
                    )));
                }
                if PREFIX {
                    pass |= _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_GT_OQ>(
                        _mm256_loadu_pd(self.key.as_ptr().add(i)),
                        _mm256_set1_pd(threshold),
                    ));
                }
                if pass != 0 {
                    return i + pass.trailing_zeros() as usize;
                }
                i += 4;
            }
        }
        self.next_candidate_scalar::<PREFIX, FORWARD, RESOURCES, NG>(
            i,
            threshold,
            candidate_r0,
            candidate_r1,
            candidate_ng,
        )
    }
}
