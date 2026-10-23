// Numerical preflight only. Rust retains label construction, cuts and insertion.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#if defined(__SSE2__) && !defined(BALDES_BATCH_FORCE_SCALAR)
#include <emmintrin.h>
#endif

struct BaldesExtensionInput {
    double resources[2];
    std::uint16_t ng;
};
struct BaldesExtensionOutput {
    double resources[2];
    std::uint8_t feasible;
};
struct BaldesExtensionArc {
    double head_lower[2], head_upper[2];
    double tail_lower[2], tail_upper[2];
    double consumption[2], jump[2];
    std::uint16_t forbidden;
    std::uint8_t has_jump;
};

template <bool Forward, bool Jump>
static void extend(const BaldesExtensionArc &arc, const BaldesExtensionInput *input,
                   BaldesExtensionOutput *output, std::size_t count) {
#if defined(__SSE2__) && !defined(BALDES_BATCH_FORCE_SCALAR)
    const __m128d hl = _mm_loadu_pd(arc.head_lower), hu = _mm_loadu_pd(arc.head_upper);
    const __m128d tl = _mm_loadu_pd(arc.tail_lower), tu = _mm_loadu_pd(arc.tail_upper);
    const __m128d consumption = _mm_loadu_pd(arc.consumption);
    const __m128d jump = _mm_loadu_pd(arc.jump);
    const __m128d infinity = _mm_set1_pd(std::numeric_limits<double>::infinity());
    const __m128d magnitude = _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL));
    const bool valid_bounds = _mm_movemask_pd(_mm_cmple_pd(hl, hu)) == 3 &&
                              (Forward || _mm_movemask_pd(_mm_cmple_pd(tl, tu)) == 3);
    for (std::size_t i = 0; i < count; ++i) {
        if (!valid_bounds || (input[i].ng & arc.forbidden)) {
            output[i].feasible = 0;
            continue;
        }
        __m128d q = _mm_loadu_pd(input[i].resources);
        if (Jump)
            q = Forward ? _mm_max_pd(q, jump) : _mm_min_pd(q, jump);
        __m128d valid;
        if (Forward) {
            q = _mm_add_pd(q, consumption);
            valid = _mm_cmplt_pd(_mm_and_pd(q, magnitude), infinity);
            q = _mm_max_pd(q, hl);
            valid = _mm_and_pd(valid, _mm_cmple_pd(q, hu));
        } else {
            valid = _mm_cmplt_pd(_mm_and_pd(q, magnitude), infinity);
            q = _mm_min_pd(q, hu);
            valid = _mm_and_pd(valid, _mm_cmple_pd(hl, q));
            q = _mm_sub_pd(q, consumption);
            valid = _mm_and_pd(valid, _mm_cmplt_pd(_mm_and_pd(q, magnitude), infinity));
            q = _mm_min_pd(q, tu);
            valid = _mm_and_pd(valid, _mm_cmple_pd(tl, q));
        }
        _mm_storeu_pd(output[i].resources, q);
        output[i].feasible = _mm_movemask_pd(valid) == 3;
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        bool feasible = (input[i].ng & arc.forbidden) == 0;
        for (int r = 0; r < 2; ++r) {
            double q = input[i].resources[r];
            if (Jump)
                q = Forward ? std::fmax(q, arc.jump[r]) : std::fmin(q, arc.jump[r]);
            if (Forward) {
                q += arc.consumption[r];
                feasible &= std::isfinite(q);
                q = std::fmax(q, arc.head_lower[r]);
                feasible &= arc.head_lower[r] <= arc.head_upper[r] && q >= arc.head_lower[r] &&
                            q <= arc.head_upper[r];
            } else {
                feasible &= std::isfinite(q);
                q = std::fmin(q, arc.head_upper[r]);
                feasible &= arc.head_lower[r] <= arc.head_upper[r] && q >= arc.head_lower[r] &&
                            q <= arc.head_upper[r];
                q -= arc.consumption[r];
                feasible &= std::isfinite(q);
                q = std::fmin(q, arc.tail_upper[r]);
                feasible &= arc.tail_lower[r] <= arc.tail_upper[r] && q >= arc.tail_lower[r] &&
                            q <= arc.tail_upper[r];
            }
            output[i].resources[r] = q;
        }
        output[i].feasible = feasible;
    }
#endif
}

// All storage is caller-owned, disjoint and valid for count elements. No
// allocation, exceptions, global state, floating-point reassociation or FMA.
extern "C" void baldes_extend_batch(const BaldesExtensionArc *arc,
                                    const BaldesExtensionInput *input,
                                    BaldesExtensionOutput *output, std::size_t count,
                                    bool forward) noexcept {
    if (forward) {
        if (arc->has_jump)
            extend<true, true>(*arc, input, output, count);
        else
            extend<true, false>(*arc, input, output, count);
    } else {
        if (arc->has_jump)
            extend<false, true>(*arc, input, output, count);
        else
            extend<false, false>(*arc, input, output, count);
    }
}
