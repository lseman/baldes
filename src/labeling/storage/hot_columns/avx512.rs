use super::*;

impl<const D: usize> HotColumns<D> {
    /// Eight-label AVX-512F kernel; comparisons retain their original EPSILON order.
    ///
    /// # Safety
    /// The caller must establish AVX-512F support on the executing CPU. Column
    /// lengths are validated here before any pointer loads.
    #[clippy::msrv = "1.89"]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn next_candidate_avx512<
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
        // SAFETY: caller establishes AVX-512F; each loop loads eight elements within
        // the checked column lengths. loadu requires no extra alignment.
        unsafe {
            let eps = _mm512_set1_pd(EPSILON);
            while i < n && n - i >= 8 {
                let mut pass: u8 = 255;
                for (r, candidate) in [candidate_r0, candidate_r1]
                    .into_iter()
                    .enumerate()
                    .take(RESOURCES)
                {
                    let stored = _mm512_loadu_pd(self.resources[r].as_ptr().add(i));
                    let c = _mm512_set1_pd(candidate);
                    let mask = if PREFIX {
                        if FORWARD {
                            _mm512_cmp_pd_mask::<_CMP_LE_OQ>(stored, _mm512_add_pd(c, eps))
                        } else {
                            _mm512_cmp_pd_mask::<_CMP_LE_OQ>(_mm512_sub_pd(c, eps), stored)
                        }
                    } else if FORWARD {
                        _mm512_cmp_pd_mask::<_CMP_LE_OQ>(c, _mm512_add_pd(stored, eps))
                    } else {
                        _mm512_cmp_pd_mask::<_CMP_LE_OQ>(_mm512_sub_pd(stored, eps), c)
                    };
                    pass &= mask;
                }
                if NG {
                    // Widen eight NG masks (16, 32 or 64 bytes) to eight u64
                    // lanes using AVX-512F (no BW/VL/DQ required).
                    let ng = self.ng.as_ptr().add(i).cast();
                    #[cfg(not(any(feature = "ng-memory-31", feature = "ng-memory-63")))]
                    let stored = _mm512_cvtepu16_epi64(_mm_loadu_si128(ng));
                    #[cfg(all(feature = "ng-memory-31", not(feature = "ng-memory-63")))]
                    let stored = _mm512_cvtepu32_epi64(_mm256_loadu_si256(ng));
                    #[cfg(feature = "ng-memory-63")]
                    let stored = _mm512_loadu_si512(ng);
                    let c = _mm512_set1_epi64(candidate_ng as i64);
                    let excluded = if PREFIX {
                        _mm512_andnot_si512(c, stored)
                    } else {
                        _mm512_andnot_si512(stored, c)
                    };
                    pass &= _mm512_cmpeq_epi64_mask(excluded, _mm512_setzero_si512());
                }
                if PREFIX {
                    pass |= _mm512_cmp_pd_mask::<_CMP_GT_OQ>(
                        _mm512_loadu_pd(self.key.as_ptr().add(i)),
                        _mm512_set1_pd(threshold),
                    );
                }
                if pass != 0 {
                    return i + pass.trailing_zeros() as usize;
                }
                i += 8;
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
