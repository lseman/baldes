/**
 * @file Hashes.h
 * @brief Custom hash and equality comparison functions for std::pair objects.
 *
 * This file defines two utility structs: `pair_hash` and `pair_equal`.
 * These are designed to allow `std::pair` objects to be used as keys
 * in unordered containers such as `std::unordered_map` or `std::unordered_set`.
 *
 * - `pair_hash`: Provides a custom hash function for hashing `std::pair` objects.
 * - `pair_equal`: Defines a comparison function for checking equality of two `std::pair` objects.
 *
 */

#pragma once

#include <functional>
#include <utility>

/**
 * @struct pair_hash_b
 * @brief A hash function object for hashing std::pair objects.
 *
 * This struct provides a custom hash function for std::pair objects,
 * allowing them to be used as keys in unordered containers such as
 * std::unordered_map or std::unordered_set.
 *
 * @tparam T1 The type of the first element in the pair.
 * @tparam T2 The type of the second element in the pair.
 */
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

/**
 * @struct pair_equal
 * @brief A functor for comparing two pairs for equality.
 *
 * This struct defines an operator() that compares two pairs of the same type
 * and returns true if both the first and second elements of the pairs are equal.
 *
 * @tparam T1 The type of the first element in the pair.
 * @tparam T2 The type of the second element in the pair.
 */
struct pair_equal {
    template <class T1, class T2>
    bool operator()(const std::pair<T1, T2> &lhs, const std::pair<T1, T2> &rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

/**
 * @brief Specialization of std::hash for std::pair<std::pair<int, int>, int>.
 *
 * This struct provides a hash function for a pair consisting of another pair of integers
 * and an integer. It combines the hash values of the inner pair and the integer to produce
 * a single hash value.
 *
 * @tparam None Template specialization for std::pair<std::pair<int, int>, int>.
 */
template <>
struct std::hash<std::pair<std::pair<int, int>, int>> {
    std::size_t operator()(const std::pair<std::pair<int, int>, int> &p) const noexcept {
        // Use pair_hash for the inner pair
        std::size_t inner_hash = pair_hash()(p.first);
        std::size_t b_hash     = std::hash<int>()(p.second);

        // Combine the hashes
        return inner_hash ^ (b_hash << 1);
    }
};