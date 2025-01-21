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
                    const auto &SRCDuals = cut_storage->SRCDuals;
                    const auto &labelSRCMap = label->SRCmap;
                    const auto &newLabelSRCMap = new_label->SRCmap;

                    simd<double> sumSRC_simd = 0;  // SIMD accumulator
                    size_t k = 0;

                    // Process in SIMD chunks
                    for (; k + simd_size <= SRCDuals.size(); k += simd_size) {
                        // Load data into SIMD registers
                        simd<double> src_duals =
                            simd<double>(&SRCDuals[k], vector_aligned);
                        simd<double> label_mod =
                            simd<double>(&labelSRCMap[k], vector_aligned);
                        simd<double> new_label_mod =
                            simd<double>(&newLabelSRCMap[k], vector_aligned);

                        // Perform comparison and create a mask
                        auto mask = label_mod > new_label_mod;

                        // Apply the mask to accumulate values conditionally
                        simd<double> masked_values = 0.0;
                        where(mask, masked_values) =
                            src_duals;  // Apply mask to src_duals
                        sumSRC_simd += masked_values;
                    }

                    // Horizontal sum of SIMD accumulator
                    double sumSRC = reduce(sumSRC_simd);

                    // Process remaining elements (scalar fallback)
                    for (; k < SRCDuals.size(); ++k) {
                        if (labelSRCMap[k] > newLabelSRCMap[k]) {
                            sumSRC += SRCDuals[k];
                        }
                    }

                    // Final check
                    if (label->cost - sumSRC > new_label->cost) continue;
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
                const auto &SRCDuals = cut_storage->SRCDuals;
                const auto &labelSRCMap = label->SRCmap;
                const auto &newLabelSRCMap = new_label->SRCmap;

                for (size_t k = 0; k < SRCDuals.size(); ++k) {
                    const auto &den = cut_storage->getCut(k).p.den;
                    const auto labelMod = labelSRCMap[k];
                    const auto newLabelMod = newLabelSRCMap[k];
                    if (labelMod > newLabelMod) {
                        sumSRC += SRCDuals[k];
                    }
                }

                if (label->cost - sumSRC > new_label->cost) continue;
            }
        }
#endif

        if (dominated) return true;
    }

    return false;
}
