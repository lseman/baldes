// Keep the gain calculation outside this kernel: only integer bitmaps are
// vectorized, preserving the order of floating-point cut evaluations.
#[inline(always)]
fn accumulate_scalar(
    bits: &[u64],
    changes: &[(i32, i32)],
    states: &[Vec<u64>],
    target: &mut [Vec<u64>],
) {
    for &(state, level) in changes {
        let source = &states[state as usize][..bits.len()];
        let destination = &mut target[level as usize][..bits.len()];
        for ((t, &b), &s) in destination.iter_mut().zip(bits).zip(source) {
            *t |= b & s;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    bits: &[u64],
    changes: &[(i32, i32)],
    states: &[Vec<u64>],
    target: &mut [Vec<u64>],
) {
    // Safe slices retain bounds checking, ordinary alignment and exclusive
    // target ownership. LLVM widens the integer loop for this target.
    accumulate_scalar(bits, changes, states, target);
}

#[inline(always)]
pub(super) fn accumulate(
    bits: &[u64],
    changes: &[(i32, i32)],
    states: &[Vec<u64>],
    target: &mut [Vec<u64>],
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if bits.len() >= 16 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection establishes the only extra precondition,
        // AVX2 support. The kernel uses safe Rust for every memory access.
        unsafe { accumulate_avx2(bits, changes, states, target) };
        return;
    }
    accumulate_scalar(bits, changes, states, target);
}

#[cfg(test)]
#[path = "../../../../tests/unit/cuts/rank_one/local_search/bitset_tests.rs"]
mod tests;
