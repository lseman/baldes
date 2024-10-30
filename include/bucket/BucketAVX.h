/**
 * @file BucketAVX.h
 * @brief SIMD-based dominance check for label comparison.
 *
 * This file contains the implementation of a SIMD-based dominance check for label comparison.
 */
#pragma once

#include "Cut.h"
#include "Definitions.h"

#include "Pools.h"

#include <experimental/simd>

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd(const Container &source, size_t start_index, size_t simd_size) {
    std::experimental::simd<T> result;
    for (size_t i = 0; i < simd_size; ++i) {
        result[i] = source[start_index + i]->cost; // Load the cost from each Label
    }
    return result;
}

/**
 * @brief Checks if a new label is dominated by any label in a given vector using SIMD operations.
 *
 * This function performs dominance checks between a new label and a vector of existing labels.
 * It utilizes SIMD operations to process multiple labels simultaneously for efficiency.
 *
 */
template <Direction D, Stage S>
inline bool check_dominance_against_vector(const Label *new_label, const std::vector<Label *> &labels,
                                           const CutStorage *cut_storage, int r_size) noexcept {
    using namespace std::experimental;
    size_t       size      = labels.size();
    const size_t simd_size = simd<double>::size(); // SIMD size based on hardware

    size_t i = 0;

    // Load the current label's cost into a SIMD register (for all lanes)
    simd<double> current_cost(new_label->cost);

    // Process the labels in SIMD batches
    for (; i + simd_size <= size; i += simd_size) {
        // Load the costs of `simd_size` labels
        simd<double> label_costs = load_simd<double>(labels, i, simd_size);

        // Compare the costs
        auto cmp_result = (label_costs <= current_cost);

        // If any label has a lower or equal cost, process further
        if (any_of(cmp_result)) {
            for (size_t j = 0; j < simd_size; ++j) {
                if (cmp_result[j]) {
                    const auto &label_resources = labels[i + j]->resources;
                    bool        dominated       = true;

                    // Perform resource checks element-wise
                    if constexpr (D == Direction::Forward) {
                        for (size_t k = 0; k < r_size; ++k) {
                            if (label_resources[k] > new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    } else if constexpr (D == Direction::Backward) {
                        for (size_t k = 0; k < r_size; ++k) {
                            if (label_resources[k] < new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    }

                    SRC_MODE_BLOCK(
                        // Additional check for Stage::Four and Stage::Enumerate
                        double sumSRC = 0.0; if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                            const auto &SRCDuals = cut_storage->SRCDuals;
                            if (!SRCDuals.empty()) {
                                const auto &labelSRCMap    = labels[i + j]->SRCmap;
                                const auto &newLabelSRCMap = new_label->SRCmap;

                                for (size_t k = 0; k < SRCDuals.size(); ++k) {
                                    const auto &den         = cut_storage->getCut(k).p.den;
                                    const auto  labelMod    = labelSRCMap[k] % den;
                                    const auto  newLabelMod = newLabelSRCMap[k] % den;
                                    if (labelMod > newLabelMod) { sumSRC += SRCDuals[k]; }
                                }
                            }

                            // SIMD cost comparison after adjusting for SRC Duals
                            if (labels[i + j]->cost - sumSRC > new_label->cost) {
                                continue; // Label is not dominated, skip
                            }
                        })

                    // Bitmap comparison for dominance
                    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                        const size_t bitmap_size = new_label->visited_bitmap.size();
                        for (size_t k = 0; k < bitmap_size; ++k) {
                            if ((labels[i + j]->visited_bitmap[k] & ~new_label->visited_bitmap[k]) != 0) {
                                dominated = false;
                                break;
                            }
                        }
                    }

                    if (dominated) {
                        return true; // Current label is dominated
                    }
                }
            }
        }
    }

    // Handle the remaining labels that couldn't be processed in the SIMD loop
    for (; i < size; ++i) {
        if (labels[i]->cost <= new_label->cost) {
            const auto &label_resources = labels[i]->resources;
            bool        dominated       = true;

            if constexpr (D == Direction::Forward) {
                for (size_t k = 0; k < r_size; ++k) {
                    if (label_resources[k] > new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            } else if constexpr (D == Direction::Backward) {
                for (size_t k = 0; k < r_size; ++k) {
                    if (label_resources[k] < new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            }

            SRC_MODE_BLOCK(
                // Check for SRC Duals (if Stage Four or Enumerate)
                double sumSRC = 0.0; if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                    const auto &SRCDuals = cut_storage->SRCDuals;
                    if (!SRCDuals.empty()) {
                        const auto &labelSRCMap    = labels[i]->SRCmap;
                        const auto &newLabelSRCMap = new_label->SRCmap;

                        for (size_t k = 0; k < SRCDuals.size(); ++k) {
                            const auto &den         = cut_storage->getCut(k).p.den;
                            const auto  labelMod    = labelSRCMap[k] % den;
                            const auto  newLabelMod = newLabelSRCMap[k] % den;
                            if (labelMod > newLabelMod) { sumSRC += SRCDuals[k]; }
                        }
                    }

                    if (labels[i]->cost - sumSRC > new_label->cost) {
                        continue; // Label is not dominated, skip
                    }
                })

            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const size_t bitmap_size = new_label->visited_bitmap.size();
                for (size_t k = 0; k < bitmap_size; ++k) {
                    if ((labels[i]->visited_bitmap[k] & ~new_label->visited_bitmap[k]) != 0) {
                        dominated = false;
                        break;
                    }
                }
            }

            if (dominated) {
                return true; // Current label is dominated
            }
        }
    }

    return false;
}
