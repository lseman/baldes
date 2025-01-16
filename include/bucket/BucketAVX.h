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

// Optimized SIMD loading
template <typename T, typename Container>
inline std::experimental::simd<T> load_simd(const Container &source,
                                            size_t start_index,
                                            size_t simd_size) {
    std::experimental::simd<T> result;
    alignas(64) std::array<T, std::experimental::simd<T>::size()> buffer;

    // Load data into a buffer for better cache locality
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] = source[start_index + i]->cost;
    }

    // Load the buffer into the SIMD register
    result.copy_from(buffer.data(), std::experimental::vector_aligned);

    return result;
}

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd_generic(const Container &source,
                                                    size_t start_index,
                                                    size_t simd_size) {
    std::experimental::simd<T> result;
    alignas(64) std::array<T, std::experimental::simd<T>::size()> buffer;

    // Load data into a buffer for better cache locality
    for (size_t i = 0; i < simd_size; ++i) {
        buffer[i] = source[start_index + i];
    }

    // Load the buffer into the SIMD register
    result.copy_from(buffer.data(), std::experimental::vector_aligned);

    return result;
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

            // Resource check based on direction
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

            // Visited nodes check for appropriate stages
            if constexpr (S == Stage::Three || S == Stage::Four ||
                          S == Stage::Enumerate) {
                for (size_t k = 0; k < label->visited_bitmap.size(); ++k) {
                    if (((label->visited_bitmap[k] &
                          new_label->visited_bitmap[k]) ^
                         label->visited_bitmap[k]) != 0) {
                        dominated = false;
                        break;
                    }
                }
            }

            if (!dominated) continue;

// SRC check for Stage Four and Enumerate
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
    }

    // Handle remaining labels
    for (; i < size; ++i) {
        const Label *label = labels[i];

        if (label->cost > new_label->cost) continue;

        // Resource check
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

        // Visited nodes check
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

// SRC check
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
