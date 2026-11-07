use super::*;

impl<'p> Individual<'p> {
    // ── Proximity / diversity ─────────────────────────────────────────────

    /// Add a reference to another individual in the proximity structure.
    pub fn add_proximity(&mut self, distance: f64, indiv_id: usize) {
        self.indivs_per_proximity
            .insert((SortableF64(distance), indiv_id));
    }

    /// Remove an individual from the proximity structure by ID.
    pub fn remove_proximity(&mut self, indiv_id: usize) {
        self.indivs_per_proximity.retain(|&(_, id)| id != indiv_id);
    }

    /// Broken pairs distance between this individual and indiv2.
    /// Measures the fraction of arcs that differ (direct or reverse).
    pub fn broken_pairs_distance(&self, indiv2: &Self) -> f64 {
        let nb_clients = self.params.nb_clients as usize;
        let differences = broken_pairs_count(
            &self.successors[1..=nb_clients],
            &self.predecessors[1..=nb_clients],
            &indiv2.successors[1..=nb_clients],
            &indiv2.predecessors[1..=nb_clients],
        );

        differences as f64 / nb_clients as f64
    }

    /// Average broken pairs distance to the `nb_closest` individuals.
    pub fn average_broken_pairs_distance_closest(&self, nb_closest: usize) -> f64 {
        let size = self.indivs_per_proximity.len().min(nb_closest);
        if size == 0 {
            return 0.0;
        }

        let sum: f64 = self
            .indivs_per_proximity
            .iter()
            .take(size)
            .map(|(d, _)| d.0)
            .sum();

        sum / size as f64
    }
}

/// Integer counts keep diversity and population ordering identical to scalar search.
fn broken_pairs_count(
    a_next: &[usize],
    a_prev: &[usize],
    b_next: &[usize],
    b_prev: &[usize],
) -> usize {
    debug_assert_eq!(a_next.len(), a_prev.len());
    debug_assert_eq!(a_next.len(), b_next.len());
    debug_assert_eq!(a_next.len(), b_prev.len());
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if a_next.len() >= 16 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected at runtime. The kernel uses bounded chunks.
        return unsafe { broken_pairs_count_avx2(a_next, a_prev, b_next, b_prev) };
    }
    broken_pairs_count_scalar(a_next, a_prev, b_next, b_prev)
}

fn broken_pairs_count_scalar(
    a_next: &[usize],
    a_prev: &[usize],
    b_next: &[usize],
    b_prev: &[usize],
) -> usize {
    let mut differences = 0;
    for j in 0..a_next.len() {
        if a_next[j] != b_next[j] && a_next[j] != b_prev[j] {
            differences += 1;
        }
        // Route starts count independently of the outgoing-arc comparison.
        if a_prev[j] == 0 && b_prev[j] != 0 && b_next[j] != 0 {
            differences += 1;
        }
    }
    differences
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn broken_pairs_count_avx2(
    a_next: &[usize],
    a_prev: &[usize],
    b_next: &[usize],
    b_prev: &[usize],
) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let mut counts = zero;
    let chunks = a_next
        .chunks_exact(4)
        .zip(a_prev.chunks_exact(4))
        .zip(b_next.chunks_exact(4))
        .zip(b_prev.chunks_exact(4));
    for (((an, ap), bn), bp) in chunks {
        // SAFETY: each chunk holds four 64-bit usize values; loads are unaligned.
        let (an, ap, bn, bp) = unsafe {
            (
                _mm256_loadu_si256(an.as_ptr().cast()),
                _mm256_loadu_si256(ap.as_ptr().cast()),
                _mm256_loadu_si256(bn.as_ptr().cast()),
                _mm256_loadu_si256(bp.as_ptr().cast()),
            )
        };
        let same_next = _mm256_or_si256(_mm256_cmpeq_epi64(an, bn), _mm256_cmpeq_epi64(an, bp));
        let b_depot = _mm256_or_si256(_mm256_cmpeq_epi64(bn, zero), _mm256_cmpeq_epi64(bp, zero));
        let missing_start = _mm256_andnot_si256(b_depot, _mm256_cmpeq_epi64(ap, zero));
        let broken = _mm256_andnot_si256(same_next, one);
        counts = _mm256_sub_epi64(_mm256_add_epi64(counts, broken), missing_start);
    }
    let mut lanes = [0usize; 4];
    // SAFETY: the destination holds four 64-bit usize values.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), counts) };
    let differences: usize = lanes.into_iter().sum();
    let tail = a_next.len() / 4 * 4;
    differences
        + broken_pairs_count_scalar(
            &a_next[tail..],
            &a_prev[tail..],
            &b_next[tail..],
            &b_prev[tail..],
        )
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/individual/diversity.rs"]
mod tests;
