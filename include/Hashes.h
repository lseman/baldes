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
#include <iomanip>
#include <sstream>
#include <utility>

/**
 * @struct pair_hash
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

/**
 * @brief Generates a hash value for a given double and index.
 *
 * This function converts a double value to a string with 17 decimal places of precision,
 * then hashes the resulting string and combines it with the provided index using a bitwise XOR operation.
 *
 * @param value The double value to be hashed.
 * @param index The index to combine with the hash of the double value.
 * @return A size_t representing the combined hash value.
 */
inline std::size_t hash_double(double value, std::size_t index) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(17) << value; // 17 decimal places precision
    std::string double_str = oss.str();
    return std::hash<std::string>{}(double_str) ^ (index * 0x9e3779b9); // Combine with index
}

/**
 * @brief Combines a hash value with an existing seed.
 *
 * This function takes an existing seed and a value, computes the hash of the value,
 * and combines it with the seed to produce a new hash value. This is useful for
 * creating composite hash values from multiple inputs.
 *
 * @tparam T The type of the value to be hashed.
 * @param seed A reference to the existing seed to be combined with the hash of the value.
 * @param value The value to be hashed and combined with the seed.
 */
template <typename T>
inline void hash_combine(std::size_t &seed, const T &value) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
