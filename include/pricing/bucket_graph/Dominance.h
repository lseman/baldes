/**
 * @file BucketAVX.h
 * @brief Optimized SIMD-based dominance check for label comparison.
 *
 */
#pragma once

#include <experimental/simd>

#include "cuts/model/Cut.h"
#include "core/Definitions.h"
#include "core/Pools.h"

using simd_double = std::experimental::native_simd<double>;

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd_generic(const Container &source, size_t start_index,
                                                    size_t simd_size) noexcept {
    constexpr size_t simd_register_size = std::experimental::simd<T>::size();

    // Fallback: Load into a buffer and then into the SIMD register
    alignas(64) std::array<T, simd_register_size> buffer = {};
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] = source[start_index + i]; // Assumes source contains raw values
    }

    return std::experimental::simd<T>(buffer.data(), std::experimental::vector_aligned);
}

template <typename T, typename Container, typename Proj>
inline std::experimental::simd<T> load_simd(const Container &source, size_t start_index, size_t simd_size,
                                            Proj proj) noexcept {
    constexpr size_t                              simd_register_size = std::experimental::simd<T>::size();
    alignas(64) std::array<T, simd_register_size> buffer             = {};

    size_t valid_size = std::min(simd_size, source.size() - start_index);
    size_t buffer_idx = 0;

    // Use the projection to load the values.
    for (size_t i = 0; i < valid_size && buffer_idx < simd_size; ++i) {
        buffer[buffer_idx++] = proj(source[start_index + i]);
    }

    // Fill remaining slots with the maximum value.
    while (buffer_idx < simd_size) { buffer[buffer_idx++] = std::numeric_limits<T>::max(); }

    return std::experimental::simd<T>(buffer.data(), std::experimental::vector_aligned);
}

// Direct SIMD load from contiguous aligned memory (no projection, no bounds check).
// Use this for SoA data that is already contiguous and aligned.
template <typename T>
inline std::experimental::simd<T> load_simd_direct(const T *source, size_t simd_size) noexcept {
    constexpr size_t simd_register_size = std::experimental::simd<T>::size();
    alignas(64) std::array<T, simd_register_size> buffer = {};
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] = source[i];
    }
    for (size_t i = simd_size; i < simd_register_size; ++i) {
        buffer[i] = std::numeric_limits<T>::max();
    }
    return std::experimental::simd<T>(buffer.data(), std::experimental::vector_aligned);
}

// Direct SIMD load for std::array (avoids container indirection).
template <typename T, size_t N>
inline std::experimental::simd<T> load_simd_direct(const std::array<T, N> &source, size_t simd_size) noexcept {
    return load_simd_direct(source.data(), simd_size);
}

template <Direction D, Stage S>
inline bool check_dominance_against_vector(const Label *__restrict__ new_label, std::span<Label *const> labels,
                                           const CutStorage *__restrict__ cut_storage,
                                           uint &__restrict__ stat_n_dom) noexcept {
    using namespace std::experimental;

    // Cache invariant data and sizes upfront
    constexpr double   tolerance  = numericutils::eps;
    const size_t       num_labels = labels.size();
    const size_t       simd_width = simd<double>::size();
    const double       new_cost   = new_label->cost;
    const simd<double> new_cost_simd(new_cost);
    const simd<double> tolerance_simd(tolerance);
    const auto        &new_resources = new_label->resources;
    const size_t       num_resources = new_resources.size();
    const auto        &new_visited   = new_label->visited_bitmap;
    const size_t       bitmap_size   = new_visited.size();

// SRC optimization - precompute if SRC is active
#ifdef SRC
    const bool   has_src = (S == Stage::Four || S == Stage::Enumerate) && cut_storage && cut_storage->activeSize() > 0;
    const auto   active_cuts   = has_src ? cut_storage->getActiveCuts() : decltype(cut_storage->getActiveCuts())();
    const int    cut_size      = has_src ? cut_storage->activeSize() : 0;
    const size_t simd_cut_size = cut_size - (cut_size % simd_width);
#endif

    // Process labels in SIMD-width chunks
    size_t i = 0;
    for (; i + simd_width <= num_labels; i += simd_width) {
        // Load all label costs in one SIMD operation
        simd<double> label_costs = load_simd<double>(labels, i, simd_width, [](const Label *lbl) { return lbl->cost; });

        // Fast raw-cost check is valid only when SRC dual compensation is not
        // active. With SRC, a label with higher raw reduced cost may still
        // dominate after subtracting applicable cut duals.
#ifdef SRC
        const auto cost_mask = has_src ? (label_costs == label_costs) : (label_costs <= (new_cost_simd + tolerance_simd));
#else
        const auto cost_mask = (label_costs <= (new_cost_simd + tolerance_simd));
#endif
        if (__builtin_expect(!any_of(cost_mask), 0)) continue;  // [[unlikely]]: most chunks pass cost check

        // For each label in this SIMD chunk that passed cost check
        for (size_t j = 0; j < simd_width; ++j) {
            if (__builtin_expect(!cost_mask[j], 1)) continue;  // [[likely]]: within-passed chunk, most pass

            const Label *label           = labels[i + j];
            if (__builtin_expect(label->is_dominated, 1)) continue;  // [[likely]]: many labels already dominated
            const auto  &label_resources = label->resources;
            bool         dominated       = true;

            // Resource check - use SIMD for larger resource sets
            if constexpr (R_SIZE >= simd_width) {
                // Process resources in SIMD-width chunks
                const size_t simd_resource_blocks    = num_resources / simd_width;
                const size_t simd_resource_remainder = num_resources % simd_width;

                // Main SIMD blocks
                for (size_t k = 0; k < simd_resource_blocks * simd_width; k += simd_width) {
                    simd<double> label_res     = load_simd_direct<double>(label_resources.data() + k, simd_width);
                    simd<double> new_label_res = load_simd_direct<double>(new_resources.data() + k, simd_width);

                    if constexpr (D == Direction::Forward) {
                        if (__builtin_expect(!all_of(label_res <= (new_label_res + tolerance_simd)), 0)) {
                            dominated = false;  // [[unlikely]]: new label dominates most candidates
                            break;
                        }
                    } else {
                        if (__builtin_expect(!all_of(label_res >= (new_label_res - tolerance_simd)), 0)) {
                            dominated = false;
                            break;
                        }
                    }
                }

                // Handle remaining resources
                if (dominated && simd_resource_remainder > 0) {
                    const size_t base_idx = simd_resource_blocks * simd_width;
                    for (size_t k = 0; k < simd_resource_remainder; ++k) {
                        const size_t idx = base_idx + k;
                        if (idx == 0) continue;
                        if constexpr (D == Direction::Forward) {
                            if (numericutils::gt(label_resources[idx], new_resources[idx])) {
                                dominated = false;
                                break;
                            }
                        } else {
                            if (numericutils::lt(label_resources[idx], new_resources[idx])) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                    if (dominated && num_resources > 0) {
                        if constexpr (D == Direction::Forward) {
                            if (numericutils::gt(label_resources[0], new_resources[0])) { dominated = false; }
                        } else {
                            if (numericutils::lt(label_resources[0], new_resources[0])) { dominated = false; }
                        }
                    }
                }
            } else {
                // Scalar processing for small resource sets - better for branch
                // prediction
                for (size_t k = 1; k < num_resources; ++k) {
                    if constexpr (D == Direction::Forward) {
                        if (__builtin_expect(numericutils::gt(label_resources[k], new_resources[k]), 0)) {
                            dominated = false;
                            break;
                        }
                    } else {
                        if (__builtin_expect(numericutils::lt(label_resources[k], new_resources[k]), 0)) {
                            dominated = false;
                            break;
                        }
                    }
                }
                if (dominated && num_resources > 0) {
                    if constexpr (D == Direction::Forward) {
                        if (__builtin_expect(numericutils::gt(label_resources[0], new_resources[0]), 0)) {
                            dominated = false;
                        }
                    } else {
                        if (__builtin_expect(numericutils::lt(label_resources[0], new_resources[0]), 0)) {
                            dominated = false;
                        }
                    }
                }
            }

            // Skip further checks if not dominated based on resources
            if (__builtin_expect(!dominated, 0)) continue;  // [[unlikely]]: new label dominates most candidates

            // Visited nodes check - only for relevant stages
            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const auto &label_visited = label->visited_bitmap;

                // Use SIMD for large bitmaps
                if constexpr (N_SIZE / 64 >= simd_width) {
                    const size_t simd_bitmap_blocks    = bitmap_size / simd_width;
                    const size_t simd_bitmap_remainder = bitmap_size % simd_width;

                    // Process bitmap in SIMD-width chunks
                    for (size_t k = 0; k < simd_bitmap_blocks * simd_width; k += simd_width) {
                        simd<uint64_t> label_vis     = load_simd_direct<uint64_t>(label_visited.data() + k, simd_width);
                        simd<uint64_t> new_label_vis = load_simd_direct<uint64_t>(new_visited.data() + k, simd_width);

                        // Check if all visited nodes in label are also in
                        // new_label
                        if (__builtin_expect(!all_of((label_vis & new_label_vis) == label_vis), 0)) {
                            dominated = false;  // [[unlikely]]: new label dominates most candidates
                            break;
                        }
                    }

                    // Handle remaining bitmap elements
                    if (dominated && simd_bitmap_remainder > 0) {
                        const size_t base_idx = simd_bitmap_blocks * simd_width;
                        for (size_t k = 0; k < simd_bitmap_remainder; ++k) {
                            const size_t idx = base_idx + k;
                            if (__builtin_expect((label_visited[idx] & new_visited[idx]) != label_visited[idx], 0)) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                } else {
                    // Scalar processing for small bitmaps
                    for (size_t k = 0; k < bitmap_size; ++k) {
                        if ((label_visited[k] & new_visited[k]) != label_visited[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }
            }

            // Skip SRC checks if not dominated based on visited nodes
            if (__builtin_expect(!dominated, 0)) continue;  // [[unlikely]]

// SRC dominance check - only for relevant stages and if SRC is active
#ifdef SRC
            if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                if (has_src) {
                    alignas(64) std::array<double, simd_width> dual_buf;
                    alignas(64) std::array<double, simd_width> label_buf;
                    alignas(64) std::array<double, simd_width> new_label_buf;

                    double       partial_sumSRC = 0.0;
                    simd<double> sumSRC_simd    = 0;
                    bool         early_break    = false;

                    // Process SRC cuts in SIMD-width chunks
                    for (size_t cut_idx = 0; cut_idx < simd_cut_size && !early_break; cut_idx += simd_width) {
                        // Prefetch next batch of cuts
                        if (cut_idx + simd_width < simd_cut_size) {
                            __builtin_prefetch(&active_cuts[cut_idx + simd_width], 0, 1);
                        }

                        // Load SRC data into aligned buffers
                        for (size_t j = 0; j < simd_width; ++j) {
                            const auto  &active_cut = active_cuts[cut_idx + j];
                            const size_t idx        = active_cut.index;
                            dual_buf[j]             = active_cut.dual_value;
                            label_buf[j]            = label->SRCmap[idx];
                            new_label_buf[j]        = new_label->SRCmap[idx];
                        }

                        // Create SIMD vectors from aligned buffers
                        simd<double> src_duals(dual_buf.data(), vector_aligned);
                        simd<double> label_mod(label_buf.data(), vector_aligned);
                        simd<double> new_label_mod(new_label_buf.data(), vector_aligned);

                        // Apply mask where label's SRC value exceeds new
                        // label's
                        auto         mask   = label_mod > new_label_mod;
                        simd<double> masked = 0.0;
                        where(mask, masked) = src_duals;
                        sumSRC_simd += masked;

                        // Check if we can early-break based on accumulated sum
                        partial_sumSRC = reduce(sumSRC_simd);
                        if (numericutils::gt(label->cost - partial_sumSRC, new_cost)) {
                            early_break = true;
                            break;
                        }
                    }

                    // Process remaining cuts scalar
                    if (!early_break) {
                        partial_sumSRC = reduce(sumSRC_simd);
                        for (size_t idx = simd_cut_size; idx < static_cast<size_t>(cut_size) && !early_break; ++idx) {
                            const auto  &active_cut = active_cuts[idx];
                            const size_t cut_idx    = active_cut.index;

                            if (label->SRCmap[cut_idx] > new_label->SRCmap[cut_idx]) {
                                partial_sumSRC += active_cut.dual_value;
                                if (numericutils::gt(label->cost - partial_sumSRC, new_cost)) {
                                    early_break = true;
                                    break;
                                }
                            }
                        }
                    }

                    // If we had an early break, this label doesn't dominate
                    if (early_break) {
                        dominated = false;
                        continue;
                    }
                }
            }
#endif

            // If we reach here and dominated is still true, we found domination
            if (dominated) {
                stat_n_dom++;
                return true;
            }
        }
    }

    // Process remaining labels using scalar code
    for (; i < num_labels; ++i) {
        const Label *label = labels[i];
        if (__builtin_expect(label->is_dominated, 1)) continue;  // [[likely]]

        // Cost check
#ifdef SRC
        if (!has_src && numericutils::gt(label->cost, new_cost)) continue;
#else
        if (numericutils::gt(label->cost, new_cost)) continue;
#endif

        bool        dominated       = true;
        const auto &label_resources = label->resources;

        // Resource check
        for (size_t k = 1; k < num_resources; ++k) {
            if constexpr (D == Direction::Forward) {
                if (__builtin_expect(numericutils::gt(label_resources[k], new_resources[k]), 0)) {
                    dominated = false;  // [[unlikely]]
                    break;
                }
            } else {
                if (__builtin_expect(numericutils::lt(label_resources[k], new_resources[k]), 0)) {
                    dominated = false;
                    break;
                }
            }
        }
        if (dominated && num_resources > 0) {
            if constexpr (D == Direction::Forward) {
                if (__builtin_expect(numericutils::gt(label_resources[0], new_resources[0]), 0)) {
                    dominated = false;
                }
            } else {
                if (__builtin_expect(numericutils::lt(label_resources[0], new_resources[0]), 0)) {
                    dominated = false;
                }
            }
        }

        if (!dominated) continue;

        // Visited nodes check
        if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
            const auto &label_visited = label->visited_bitmap;
            for (size_t k = 0; k < bitmap_size; ++k) {
                if ((label_visited[k] & new_visited[k]) != label_visited[k]) {
                    dominated = false;
                    break;
                }
            }
        }

        if (!dominated) continue;

// SRC check
#ifdef SRC
        if constexpr (S == Stage::Four || S == Stage::Enumerate) {
            if (has_src) {
                double sumSRC      = 0.0;
                bool   early_break = false;

                for (int cut_idx = 0; cut_idx < cut_size && !early_break; ++cut_idx) {
                    const auto  &active_cut = active_cuts[cut_idx];
                    const size_t idx        = active_cut.index;

                    if (label->SRCmap[idx] > new_label->SRCmap[idx]) {
                        sumSRC += active_cut.dual_value;
                        if (numericutils::gt(label->cost - sumSRC, new_cost)) {
                            early_break = true;
                            break;
                        }
                    }
                }

                if (early_break) {
                    dominated = false;
                    continue;
                }
            }
        }
#endif

        // If we reach here and dominated is still true, we found domination
        if (dominated) {
            stat_n_dom++;
            return true;
        }
    }

    // No domination found
    return false;
}

/**
 * @brief Batch dominance check: test B new_labels against the same set of existing labels.
 *
 * Uses SIMD to compare multiple new_labels' costs/resources/visited against each
 * existing label in parallel. Returns true if ANY new_label is dominated by an
 * existing label.
 *
 * Batch size is encoded in the template parameter BATCH_SIZE (typically 4 or 8).
 */
template <Direction D, Stage S, size_t BATCH_SIZE>
inline bool check_dominance_batch(const Label *__restrict__ batch_labels[BATCH_SIZE],
                                  std::span<Label *const> labels,
                                  const CutStorage *__restrict__ cut_storage,
                                  uint &__restrict__ stat_n_dom) noexcept {
    using namespace std::experimental;

    constexpr double   tolerance  = numericutils::eps;
    const size_t       num_labels = labels.size();
    const size_t       simd_width = simd<double>::size();

    // Pre-load batch costs and resources into SIMD-friendly arrays.
    alignas(64) std::array<double, BATCH_SIZE> batch_costs;
    alignas(64) std::array<std::array<double, R_SIZE>, BATCH_SIZE> batch_resources;
    alignas(64) std::array<std::array<uint64_t, Label::bitmap_words>, BATCH_SIZE> batch_visited;

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        if (!batch_labels[b]) {
            batch_costs[b] = std::numeric_limits<double>::max();
            continue;
        }
        batch_costs[b] = batch_labels[b]->cost;
        for (size_t r = 0; r < R_SIZE; ++r) batch_resources[b][r] = batch_labels[b]->resources[r];
        for (size_t w = 0; w < Label::bitmap_words; ++w) batch_visited[b][w] = batch_labels[b]->visited_bitmap[w];
    }

    // Process existing labels in SIMD-width chunks
    for (size_t i = 0; i < num_labels; ++i) {
        const Label *label = labels[i];
        if (__builtin_expect(label->is_dominated, 1)) continue;

        // Cost check: existing label dominates new_label b if label->cost <= batch_costs[b]
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            if (!batch_labels[b]) continue;
            if (numericutils::gt(label->cost, batch_costs[b])) continue;

            bool dominated = true;

            // Resource check
            if constexpr (R_SIZE >= 2) {
                for (size_t r = 1; r < R_SIZE; ++r) {
                    if constexpr (D == Direction::Forward) {
                        if (__builtin_expect(numericutils::gt(label->resources[r], batch_resources[b][r]), 0)) {
                            dominated = false; break;
                        }
                    } else {
                        if (__builtin_expect(numericutils::lt(label->resources[r], batch_resources[b][r]), 0)) {
                            dominated = false; break;
                        }
                    }
                }
            }
            if (dominated && R_SIZE > 0) {
                if constexpr (D == Direction::Forward) {
                    if (__builtin_expect(numericutils::gt(label->resources[0], batch_resources[b][0]), 0)) {
                        dominated = false;
                    }
                } else {
                    if (__builtin_expect(numericutils::lt(label->resources[0], batch_resources[b][0]), 0)) {
                        dominated = false;
                    }
                }
            }
            if (!dominated) continue;

            // Visited check
            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                for (size_t w = 0; w < Label::bitmap_words; ++w) {
                    if ((label->visited_bitmap[w] & batch_visited[b][w]) != batch_visited[b][w]) {
                        dominated = false; break;
                    }
                }
            }
            if (!dominated) continue;

            // If we reach here, existing label dominates batch_labels[b]
            stat_n_dom++;
            return true;
        }
    }

    return false;
}
