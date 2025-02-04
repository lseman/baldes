/**
 * @file BucketAVX.h
 * @brief Optimized SIMD-based dominance check for label comparison.
 */
#pragma once

#include <experimental/simd>

#include "Cut.h"
#include "Definitions.h"
#include "Pools.h"

namespace {
constexpr double EPSILON = -1e-3;
constexpr size_t UNROLL_SIZE = 4;
}  // namespace

using simd_double = std::experimental::native_simd<double>;
using simd_abi = std::experimental::simd_abi::native<double>;

// Helper function for loading SIMD with correct type
inline simd_double load_simd_values(const std::span<const double> &source,
                                    size_t start_index) {
    constexpr size_t simd_register_size = simd_double::size();
    alignas(64) std::array<double, simd_register_size> buffer = {};

    for (size_t i = 0; i < simd_register_size; ++i) {
        buffer[i] = source[start_index + i];
    }

    return simd_double(buffer.data(), std::experimental::vector_aligned);
}

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd(const Container &source,
                                            size_t start_index,
                                            size_t simd_size) {
    constexpr size_t simd_register_size = std::experimental::simd<T>::size();

    // Fallback: Load into a buffer and then into the SIMD register
    alignas(64) std::array<T, simd_register_size> buffer = {};
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] =
            source[start_index + i]->cost;  // Assumes source contains pointers
    }

    return std::experimental::simd<T>(buffer.data(),
                                      std::experimental::vector_aligned);
}

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd_generic(const Container &source,
                                                    size_t start_index,
                                                    size_t simd_size) {
    constexpr size_t simd_register_size = std::experimental::simd<T>::size();

    // Fallback: Load into a buffer and then into the SIMD register
    alignas(64) std::array<T, simd_register_size> buffer = {};
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] =
            source[start_index + i];  // Assumes source contains raw values
    }

    return std::experimental::simd<T>(buffer.data(),
                                      std::experimental::vector_aligned);
}
template <Direction D, Stage S>
inline bool check_dominance_against_vector(const Label *new_label,
                                           const std::vector<Label *> &labels,
                                           const CutStorage *cut_storage,
                                           int r_size) noexcept {
    using namespace std::experimental;
    const size_t size = labels.size();
    const size_t simd_size = simd<double>::size();

    // Cache frequently accessed data
    const double new_label_cost = new_label->cost;
    simd<double> current_cost(new_label_cost);

    // Pre-allocated SIMD buffers
    alignas(64) std::array<double, simd<double>::size()> resources_buffer;
    alignas(64) std::array<double, simd<double>::size()> new_resources_buffer;

    // Process labels in SIMD-width chunks
    size_t i = 0;
    for (; i + simd_size <= size; i += simd_size) {
        // Load and compare costs
        simd<double> label_costs = load_simd<double>(labels, i, simd_size);
        auto cost_mask = (label_costs <= current_cost);

        if (!any_of(cost_mask)) continue;

        // Check each label in the SIMD chunk that passed the cost check
        for (size_t j = 0; j < simd_size; ++j) {
            if (!cost_mask[j]) continue;

            const Label *label = labels[i + j];

            // Resource check
            bool dominated = true;
            if constexpr (R_SIZE >= simd_size) {
                // Use SIMD for large resource sizes
                for (size_t k = 0; k < new_label->resources.size();
                     k += simd_size) {
                    simd<double> label_resources = load_simd_generic<double>(
                        label->resources, k, simd_size);
                    simd<double> new_label_resources =
                        load_simd_generic<double>(new_label->resources, k,
                                                  simd_size);

                    if constexpr (D == Direction::Forward) {
                        auto resource_mask =
                            (label_resources <= new_label_resources);
                        if (!all_of(resource_mask)) {
                            dominated = false;
                            break;
                        }
                    } else if constexpr (D == Direction::Backward) {
                        auto resource_mask =
                            (label_resources >= new_label_resources);
                        if (!all_of(resource_mask)) {
                            dominated = false;
                            break;
                        }
                    }
                }
            } else {
                // Use scalar for small resource sizes
                for (size_t k = 0; k < new_label->resources.size(); ++k) {
                    if constexpr (D == Direction::Forward) {
                        if (label->resources[k] > new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    } else if constexpr (D == Direction::Backward) {
                        if (label->resources[k] < new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }
            }

            if (!dominated) continue;

            // Visited nodes check
            if constexpr (S == Stage::Three || S == Stage::Four ||
                          S == Stage::Enumerate) {
                if constexpr (N_SIZE / 64 >= simd_size) {
                    // Use SIMD for large bitmap sizes
                    for (size_t k = 0; k < label->visited_bitmap.size();
                         k += simd_size) {
                        simd<uint64_t> label_visited =
                            load_simd_generic<uint64_t>(label->visited_bitmap,
                                                        k, simd_size);
                        simd<uint64_t> new_label_visited =
                            load_simd_generic<uint64_t>(
                                new_label->visited_bitmap, k, simd_size);

                        auto visited_mask =
                            ((label_visited & new_label_visited) ^
                             label_visited) == 0;
                        if (!all_of(visited_mask)) {
                            dominated = false;
                            break;
                        }
                    }
                } else {
                    // Use scalar for small bitmap sizes
                    for (size_t k = 0; k < label->visited_bitmap.size(); ++k) {
                        if (((label->visited_bitmap[k] &
                              new_label->visited_bitmap[k]) ^
                             label->visited_bitmap[k]) != 0) {
                            dominated = false;
                            break;
                        }
                    }
                }
            }

            if (!dominated) continue;

#ifdef SRC
            if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                if (cut_storage && !cut_storage->SRCDuals.empty()) {
                    using namespace std::experimental;
                    const size_t simd_size = simd<double>::size();
                    const auto active_cuts = cut_storage->getActiveCuts();
                    const auto cut_size = cut_storage->activeSize();
                    const size_t simd_cut_size =
                        cut_size - (cut_size % simd_size);

                    // Prepare aligned buffers for SIMD processing
                    alignas(64) std::array<double, simd_size> dual_buffer;
                    alignas(64) std::array<double, simd_size> label_buffer;
                    alignas(64) std::array<double, simd_size> new_label_buffer;

                    double partial_sumSRC = 0.0;   // Track partial sum
                    simd<double> sumSRC_simd = 0;  // SIMD accumulator
                    bool should_terminate = false;

                    // Process in SIMD chunks
                    for (size_t base_idx = 0;
                         base_idx < simd_cut_size && !should_terminate;
                         base_idx += simd_size) {
                        // Fill buffers with active cut data
                        for (size_t j = 0; j < simd_size; ++j) {
                            const auto &active_cut = active_cuts[base_idx + j];
                            const size_t idx = active_cut.index;
                            dual_buffer[j] = active_cut.dual_value;
                            label_buffer[j] = label->SRCmap[idx];
                            new_label_buffer[j] = new_label->SRCmap[idx];
                        }

                        // Load data into SIMD registers
                        simd<double> src_duals =
                            simd<double>(dual_buffer.data(), vector_aligned);
                        simd<double> label_mod =
                            simd<double>(label_buffer.data(), vector_aligned);
                        simd<double> new_label_mod = simd<double>(
                            new_label_buffer.data(), vector_aligned);

                        // Perform comparison and create a mask
                        auto mask = label_mod > new_label_mod;

                        // Apply the mask to accumulate values conditionally
                        simd<double> masked_values = 0.0;
                        where(mask, masked_values) =
                            src_duals;  // Apply mask to src_duals
                        sumSRC_simd += masked_values;

                        // Check intermediary result every SIMD_CHECK_INTERVAL
                        // chunks
                        // Get current partial sum
                        partial_sumSRC = reduce(sumSRC_simd);

                        // Early termination check
                        if (label->cost - partial_sumSRC > new_label->cost) {
                            should_terminate = true;
                            break;
                        }
                    }

                    if (!should_terminate) {
                        // Final SIMD sum
                        partial_sumSRC = reduce(sumSRC_simd);

                        // Process remaining elements (scalar fallback)
                        for (size_t i = simd_cut_size;
                             i < cut_size && !should_terminate; ++i) {
                            const auto &active_cut = active_cuts[i];
                            const size_t idx = active_cut.index;
                            if (label->SRCmap[idx] > new_label->SRCmap[idx]) {
                                partial_sumSRC += active_cut.dual_value;

                                // Check after each scalar addition
                                if (label->cost - partial_sumSRC >
                                    new_label->cost) {
                                    should_terminate = true;
                                    break;
                                }
                            }
                        }
                    }

                    // Final check that will affect the outer scope
                    if (should_terminate ||
                        label->cost - partial_sumSRC > new_label->cost) {
                        continue;
                    }
                }
            }
#endif
            if (dominated) return true;
        }
    }

    // Handle remaining labels (scalar fallback)
    for (; i < size; ++i) {
        const Label *label = labels[i];

        if (label->cost > new_label->cost) continue;

        // Resource check (scalar)
        bool dominated = true;
        for (size_t k = 0; k < new_label->resources.size(); ++k) {
            if constexpr (D == Direction::Forward) {
                if (label->resources[k] > new_label->resources[k]) {
                    dominated = false;
                    break;
                }
            } else if constexpr (D == Direction::Backward) {
                if (label->resources[k] < new_label->resources[k]) {
                    dominated = false;
                    break;
                }
            }
        }

        if (!dominated) continue;

        // Visited nodes check (scalar)
        if constexpr (S == Stage::Three || S == Stage::Four ||
                      S == Stage::Enumerate) {
            for (size_t k = 0; k < label->visited_bitmap.size(); ++k) {
                if (((label->visited_bitmap[k] & new_label->visited_bitmap[k]) ^
                     label->visited_bitmap[k]) != 0) {
                    dominated = false;
                    break;
                }
            }
        }

        if (!dominated) continue;

// SRC check (scalar)
#ifdef SRC
        if constexpr (S == Stage::Four || S == Stage::Enumerate) {
            if (cut_storage && !cut_storage->SRCDuals.empty()) {
                double sumSRC = 0;
                const auto active_cuts = cut_storage->getActiveCuts();

                for (const auto &active_cut : active_cuts) {
                    const size_t idx = active_cut.index;
                    const double dual_value = active_cut.dual_value;
                    const auto &cut = *active_cut.cut_ptr;

                    // Access SRCmap values using the index from active cut
                    const auto labelMod = label->SRCmap[idx];
                    const auto newLabelMod = new_label->SRCmap[idx];

                    if (labelMod > newLabelMod) {
                        sumSRC += dual_value;
                        if (label->cost - sumSRC > new_label->cost) {
                            break;
                        }
                    }
                }
            }
        }
#endif

        if (dominated) return true;
    }

    return false;
}
