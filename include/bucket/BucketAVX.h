/**
 * @file BucketAVX.h
 * @brief Optimized SIMD-based dominance check for label comparison.
 */
#pragma once

#include "Cut.h"
#include "Definitions.h"
#include "Pools.h"
#include <experimental/simd>

namespace {
    constexpr double EPSILON = -1e-3;
    constexpr size_t UNROLL_SIZE = 4;
}

// Optimized SIMD loading
template <typename T, typename Container>
inline std::experimental::simd<T> load_simd(const Container &source, size_t start_index, size_t simd_size) {
    std::experimental::simd<T> result;
    for (size_t i = 0; i < simd_size; ++i) {
        result[i] = source[start_index + i]->cost;
    }
    return result;
}

template <typename T, typename Container>
inline std::experimental::simd<T> load_simd_generic(const Container &source, size_t start_index, size_t simd_size) {
    std::experimental::simd<T> result;
    for (size_t i = 0; i < simd_size; ++i) {
        result[i] = source[start_index + i];
    }
    return result;
}

template <Direction D, Stage S>
inline bool check_dominance_against_vector(const Label *new_label, const std::vector<Label *> &labels,
                                         const CutStorage *cut_storage, int r_size) noexcept {
    using namespace std::experimental;
    const size_t size = labels.size();
    const size_t simd_size = simd<double>::size();

    // Cache frequently accessed data
    const double new_label_cost = new_label->cost;
    simd<double> current_cost(new_label_cost);
    
    // Pre-allocated SIMD buffers
    alignas(32) std::array<double, simd<double>::size()> resources_buffer;
    alignas(32) std::array<double, simd<double>::size()> new_resources_buffer;

    size_t i = 0;
    for (; i + simd_size <= size; i += simd_size) {
        simd<double> label_costs = load_simd<double>(labels, i, simd_size);
        auto cost_mask = (label_costs <= current_cost);

        if (!any_of(cost_mask)) continue;

        for (size_t j = 0; j < simd_size; ++j) {
            if (!cost_mask[j]) continue;

            const Label *current_label = labels[i + j];
            bool dominated = true;

            // Resource check optimization
            if constexpr (D == Direction::Forward) {
                if (r_size >= simd_size) {
                    size_t k = 0;
                    for (; k + simd_size <= r_size; k += simd_size) {
                        // Optimized resource loading
                        for (size_t m = 0; m < simd_size; ++m) {
                            resources_buffer[m] = current_label->resources[k + m];
                            new_resources_buffer[m] = new_label->resources[k + m];
                        }

                        simd<double> res_vec = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        simd<double> new_res_vec = load_simd_generic<double>(new_resources_buffer.data(), 0, simd_size);

                        if (any_of(res_vec > new_res_vec)) {
                            dominated = false;
                            break;
                        }
                    }

                    if (dominated) {
                        for (; k < r_size; ++k) {
                            if (current_label->resources[k] > new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                } else {
                    for (size_t k = 0; k < r_size; ++k) {
                        if (current_label->resources[k] > new_label->resources[k]) {
                            dominated = false;
                            break;
                        }
                    }
                }
            } else {
                if (r_size >= simd_size) {
                    size_t k = 0;
                    for (; k + simd_size <= r_size; k += simd_size) {
                        for (size_t m = 0; m < simd_size; ++m) {
                            resources_buffer[m] = current_label->resources[k + m];
                            new_resources_buffer[m] = new_label->resources[k + m];
                        }

                        simd<double> res_vec = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        simd<double> new_res_vec = load_simd_generic<double>(new_resources_buffer.data(), 0, simd_size);

                        if (any_of(res_vec < new_res_vec)) {
                            dominated = false;
                            break;
                        }
                    }

                    if (dominated) {
                        for (; k < r_size; ++k) {
                            if (current_label->resources[k] < new_label->resources[k]) {
                                dominated = false;
                                break;
                            }
                        }
                    }
                } else {
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
                    const auto &labelSRCMap = current_label->SRCmap;
                    const auto &newLabelSRCMap = new_label->SRCmap;
                    const size_t src_size = SRCDuals.size();
                    size_t k = 0;

                    for (; k + simd_size <= src_size; k += simd_size) {
                        for (size_t m = 0; m < simd_size; ++m) {
                            resources_buffer[m] = 0.0;
                            const double dual = SRCDuals[k + m];
                            if (dual <= EPSILON && labelSRCMap[k + m] > newLabelSRCMap[k + m]) {
                                resources_buffer[m] = dual;
                            }
                        }

                        simd<double> src_vec = load_simd_generic<double>(resources_buffer.data(), 0, simd_size);
                        sumSRC += reduce(src_vec);
                    }

                    for (; k < src_size; ++k) {
                        const double dual = SRCDuals[k];
                        if (dual <= EPSILON && labelSRCMap[k] > newLabelSRCMap[k]) {
                            sumSRC += dual;
                        }
                    }
                }

                if (current_label->cost - sumSRC > new_label->cost) continue;
            })

            if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
                const size_t bitmap_size = new_label->visited_bitmap.size();
                dominated = true;

                size_t k = 0;
                constexpr size_t chunk_size = sizeof(size_t) * 8;

                for (; k + chunk_size <= bitmap_size; k += chunk_size) {
                    const size_t chunk1 = *reinterpret_cast<const size_t *>(&current_label->visited_bitmap[k]);
                    const size_t chunk2 = *reinterpret_cast<const size_t *>(&new_label->visited_bitmap[k]);

                    if (((chunk1 & chunk2) ^ chunk1) != 0) {
                        dominated = false;
                        break;
                    }
                }

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

    // Handle remaining labels
    for (; i < size; ++i) {
        if (labels[i]->cost <= new_label_cost) {
            const auto &label_resources = labels[i]->resources;
            bool dominated = true;

            if constexpr (D == Direction::Forward) {
                for (size_t k = 0; k < r_size; ++k) {
                    if (label_resources[k] > new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            } else {
                for (size_t k = 0; k < r_size; ++k) {
                    if (label_resources[k] < new_label->resources[k]) {
                        dominated = false;
                        break;
                    }
                }
            }

            if (!dominated) continue;

            SRC_MODE_BLOCK(
                double sumSRC = 0.0; if constexpr (S == Stage::Four || S == Stage::Enumerate) {
                    const auto &SRCDuals = cut_storage->SRCDuals;
                    if (!SRCDuals.empty()) {
                        const auto &labelSRCMap = labels[i]->SRCmap;
                        const auto &newLabelSRCMap = new_label->SRCmap;
                        sumSRC = std::transform_reduce(
                            SRCDuals.begin(), SRCDuals.end(), 0.0, std::plus<>(),
                            [&](const auto &dual) {
                                size_t k = &dual - &SRCDuals[0];
                                if (dual > EPSILON) return 0.0;
                                return (labelSRCMap[k] > newLabelSRCMap[k]) ? dual : 0.0;
                            });
                    }

                    if (labels[i]->cost - sumSRC > new_label->cost) continue;
                }
            )

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

            if (dominated) return true;
        }
    }

    return false;
}