#pragma once

#include "Definitions.h"

#include "DataClasses.h"

#include <experimental/simd>

template <Direction D, Stage S>
inline bool check_dominance_against_vector(Label *&new_label, const std::vector<Label *> &labels) noexcept {
    using namespace std::experimental;
    size_t       size      = labels.size();
    const size_t simd_size = simd<double>::size(); // SIMD size based on hardware

    size_t i = 0;

    // Load the current label's cost into a SIMD register (for all lanes)
    simd<double> current_cost(new_label->cost);

    // Process the labels in SIMD batches
    for (; i + simd_size <= size; i += simd_size) {
        // Load the costs of `simd_size` labels
        simd<double> label_costs;
        for (std::size_t j = 0; j < simd_size; ++j) {
            label_costs[j] = labels[i + j]->cost; // Load individual elements
        }
        // Compare the costs
        auto cmp_result = (label_costs <= current_cost);

        // If any label has a lower or equal cost, process further
        if (any_of(cmp_result)) {
            for (size_t j = 0; j < simd_size; ++j) {
                if (cmp_result[j]) {
                    const auto &label_resources = labels[i + j]->resources;

                    bool dominated = true;

                    // Perform resource checks element-wise
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

                    // Bitmap comparison for dominance
                    if (dominated && (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate)) {
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
