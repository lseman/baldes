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

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd_generic(const Container &source, size_t start_index, size_t simd_size) {
    std::experimental::simd<T> result;
    for (size_t i = 0; i < simd_size; ++i) {
        result[i] = source[start_index + i]; // Load the cost from each Label
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
    const size_t size      = labels.size();
    const size_t simd_size = simd<double>::size();

    // Pre-load new_label data into SIMD registers for comparison
    simd<double> current_cost(new_label->cost);

    // For resource checks
    alignas(32) std::array<double, simd_size> resources_buffer;
    alignas(32) std::array<double, simd_size> new_resources_buffer;

    size_t i = 0;
    for (; i + simd_size <= size; i += simd_size) {
        // Load costs vectorized
        simd<double> label_costs = load_simd<double>(labels, i, simd_size);
        auto         cost_mask   = (label_costs <= current_cost);

        if (!any_of(cost_mask)) continue;

        for (size_t j = 0; j < simd_size; ++j) {
            if (!cost_mask[j]) continue;

            const Label *current_label = labels[i + j];
            bool         dominated     = true;

            // Resource check optimization
            if constexpr (D == Direction::Forward) {
                // Use SIMD for resource comparison when r_size is big enough
                if (r_size >= simd_size) {
                    size_t k = 0;
                    for (; k + simd_size <= r_size; k += simd_size) {
                        // Load resources into SIMD registers
                        for (size_t m = 0; m < simd_size; ++m) {
                            resources_buffer[m]     = current_label->resources[k + m];
                            new_resources_buffer[m] = new_label->resources[k + m];
                        }

                        simd<double> res_vec     = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        simd<double> new_res_vec = load_simd_generic<double>(new_resources_buffer.data(), 0, simd_size);

                        if (any_of(res_vec > new_res_vec)) {
                            dominated = false;
                            break;
                        }
                    }

                    // Handle remaining resources
                    if (dominated) {
                        for (; k < r_size; ++k) {
                            if (current_label->resources[k] > new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                } else {
                    // Original scalar code for small r_size
                    for (size_t k = 0; k < r_size; ++k) {
                        if (current_label->resources[k] > new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }
            } else {
                // Similar SIMD optimization for Backward direction
                if (r_size >= simd_size) {
                    size_t k = 0;
                    for (; k + simd_size <= r_size; k += simd_size) {
                        // Load resources into SIMD registers
                        for (size_t m = 0; m < simd_size; ++m) {
                            resources_buffer[m]     = current_label->resources[k + m];
                            new_resources_buffer[m] = new_label->resources[k + m];
                        }

                        simd<double> res_vec     = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        simd<double> new_res_vec = load_simd_generic<double>(new_resources_buffer.data(), 0, simd_size);

                        if (any_of(res_vec < new_res_vec)) {
                            dominated = false;
                            break;
                        }
                    }

                    // Handle remaining resources
                    if (dominated) {
                        for (; k < r_size; ++k) {
                            if (current_label->resources[k] < new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                } else {
                    // Original scalar code for small r_size
                    for (size_t k = 0; k < r_size; ++k) {
                        if (current_label->resources[k] < new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }
            }

            if (!dominated) continue;

            SRC_MODE_BLOCK(double sumSRC = 0.0; if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                const auto &SRCDuals = cut_storage->SRCDuals;
                if (!SRCDuals.empty()) {
                    const auto &labelSRCMap    = current_label->SRCmap;
                    const auto &newLabelSRCMap = new_label->SRCmap;

                    // Process SRCDuals in SIMD chunks
                    const size_t src_size = SRCDuals.size();
                    size_t       k        = 0;

                    for (; k + simd_size <= src_size; k += simd_size) {
                        // Load SRCDuals and maps into buffers
                        for (size_t m = 0; m < simd_size; ++m) {
                            const double dual = SRCDuals[k + m];
                            resources_buffer[m] =
                                (dual <= -1e-3 && labelSRCMap[k + m] > newLabelSRCMap[k + m]) ? dual : 0.0;
                        }

                        simd<double> src_vec = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        sumSRC += std::experimental::reduce(src_vec);
                    }

                    // Handle remaining elements
                    for (; k < src_size; ++k) {
                        const double dual = SRCDuals[k];
                        if (dual <= -1e-3 && labelSRCMap[k] > newLabelSRCMap[k]) { sumSRC += dual; }
                    }
                }

                if (current_label->cost - sumSRC > new_label->cost) { continue; }
            })

            // Bitmap comparison optimization for relevant stages
            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const size_t bitmap_size = new_label->visited_bitmap.size();
                dominated                = true;

                // Process bitmap in chunks when possible
                size_t           k          = 0;
                constexpr size_t chunk_size = sizeof(size_t) * 8; // Process whole words at a time

                for (; k + chunk_size <= bitmap_size; k += chunk_size) {
                    size_t chunk1 = *reinterpret_cast<const size_t *>(&current_label->visited_bitmap[k]);
                    size_t chunk2 = *reinterpret_cast<const size_t *>(&new_label->visited_bitmap[k]);

                    if (((chunk1 & chunk2) ^ chunk1) != 0) {
                        dominated = false;
                        break;
                    }
                }

                // Handle remaining bits
                if (dominated && k < bitmap_size) {
                    for (; k < bitmap_size; ++k) {
                        if (((current_label->visited_bitmap[k] & new_label->visited_bitmap[k]) ^
                             current_label->visited_bitmap[k]) != 0) {
                            dominated = false;
                            break;
                        }
                    }
                }
            }

            if (dominated) return true;
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
                        sumSRC                     = std::transform_reduce(
                            SRCDuals.begin(), SRCDuals.end(), 0.0, std::plus<>(), [&](const auto &dual) {
                                size_t k = &dual - &SRCDuals[0];  // Get the index from the iterator
                                if (dual > -1e-3) { return 0.0; } // Skip non-SRC cuts
                                return (labelSRCMap[k] > newLabelSRCMap[k]) ? dual : 0.0;
                            });
                    }

                    if (labels[i]->cost - sumSRC > new_label->cost) {
                        continue; // Label is not dominated, skip
                    }
                })

            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const size_t bitmap_size = new_label->visited_bitmap.size();
                for (size_t k = 0; k < bitmap_size; ++k) {
                    if (((labels[i]->visited_bitmap[k] & new_label->visited_bitmap[k]) ^
                         labels[i]->visited_bitmap[k]) != 0) {
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
