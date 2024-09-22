#pragma once

#include "Definitions.h"

template <Direction D, Stage S>
inline bool check_dominance_against_vector(Label *&new_label, const std::vector<Label *> &labels) noexcept {
    size_t       size      = labels.size();
    const size_t simd_size = 4; // Process 4 labels at a time with AVX
    size_t       i         = 0;

    // Load the current label's cost into a SIMD register (4 copies of the current label's cost)
    __m256d current_cost = _mm256_set1_pd(new_label->cost); // Assuming "this" is the current Label

    // Process the labels 4 at a time
    for (; i + simd_size <= size; i += simd_size) {
        // Load the costs of 4 labels into an AVX register
        __m256d label_costs =
            _mm256_set_pd(labels[i + 3]->cost, labels[i + 2]->cost, labels[i + 1]->cost, labels[i]->cost);

        // Compare the costs: check if any label has a lower or equal cost than the current label
        __m256d cmp_result = _mm256_cmp_pd(label_costs, current_cost, _CMP_LE_OQ);

        // Extract the comparison results as a mask (4 bits for 4 comparisons)
        int mask = _mm256_movemask_pd(cmp_result);

        if (mask == 0) continue; // Skip if no label has a lower or equal cost

        // Process labels that passed the cost check
        for (size_t j = 0; j < simd_size; ++j) {
            if (mask & (1 << j)) {
                const auto &label_resources = labels[i + j]->resources;

                bool dominated = true;

                // Perform resource checks outside of SIMD based on direction
                if constexpr (D == Direction::Forward) {
                    for (size_t k = 0; k < new_label->resources.size(); ++k) {
                        // Check if any resource fails the dominance condition
                        if (label_resources[k] > new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                } else if constexpr (D == Direction::Backward) {
                    for (size_t k = 0; k < new_label->resources.size(); ++k) {
                        if (label_resources[k] < new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }

                if (!dominated) continue; // Skip if not dominated

                // Optimized bitmap comparison with pre-check outside the SIMD loop
                if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                    const size_t bitmap_size      = new_label->visited_bitmap.size();
                    const size_t simd_bitmap_size = 4;
                    size_t       k                = 0;

                    for (; k + simd_bitmap_size <= bitmap_size; k += simd_bitmap_size) {
                        __m256i label_bitmap =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&labels[i + j]->visited_bitmap[k]));
                        __m256i current_bitmap =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&new_label->visited_bitmap[k]));

                        // Perform the bitmap comparison: label's bitmap must be a subset of the current bitmap
                        __m256i result = _mm256_andnot_si256(current_bitmap, label_bitmap);

                        if (!_mm256_testz_si256(result, result)) {
                            dominated = false;
                            break;
                        }
                    }

                    // Handle remaining non-SIMD bitmap elements
                    for (; k < bitmap_size; ++k) {
                        if ((labels[i + j]->visited_bitmap[k] & ~new_label->visited_bitmap[k]) != 0) {
                            dominated = false;
                            break;
                        }
                    }
                }

                if (dominated) {
                    return true; // Current label is dominated, exit early
                }
            }
        }
    }

    // Handle remaining labels that couldn't be processed in multiples of 4
    for (; i < size; ++i) {
        if (labels[i]->cost <= new_label->cost) {
            const auto &label_resources = labels[i]->resources;

            bool dominated = true;

            if constexpr (D == Direction::Forward) {
                for (size_t k = 0; k < new_label->resources.size(); ++k) {
                    if (label_resources[k] > new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            } else if constexpr (D == Direction::Backward) {
                for (size_t k = 0; k < new_label->resources.size(); ++k) {
                    if (label_resources[k] < new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            }

            if (!dominated) continue;

            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const size_t bitmap_size = new_label->visited_bitmap.size();
                for (size_t k = 0; k < bitmap_size; ++k) {
                    if ((labels[i]->visited_bitmap[k] & ~new_label->visited_bitmap[k]) != 0) {
                        dominated = false;
                        break;
                    }
                }
            }

            if (dominated) { return true; }
        }
    }

    return false;
}
