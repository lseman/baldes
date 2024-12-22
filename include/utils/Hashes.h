/*
 * @file Hashes.h
 * @brief Hash functions for various data types and structures.
 *
 * This file contains hash functions for various data types and structures, including:
 * - Hash functions for vectors of integers and doubles.
 * - Hash functions for pairs and unordered sets.
 * - Specialized hash functions for specific data types.
 *
 */

#pragma once

#include <bit> // for std::bit_cast (C++20)
#include <cstdint>
#include <functional>
#include <utility>  // for std::pair
#include <xxhash.h> // Include XXHash header

/**
 * @brief Combines hash values for multiple values using C++20, without recursion.
 */
template <typename T, typename... Rest>
constexpr void hash_combine(std::size_t &seed, const T &value, const Rest &...rest) noexcept {
    // Use XXH3 for faster hashing
    seed ^= XXH3_64bits_withSeed(&value, sizeof(T), seed);
    (..., (seed ^= XXH3_64bits_withSeed(&rest, sizeof(Rest), seed)));
}

/**
 * @struct arc_map_hash
 * @brief Optimized hash for std::pair<std::pair<int, int>, int>.
 */
struct arc_map_hash {
    constexpr std::size_t operator()(const std::pair<std::pair<int, int>, int> &p) const noexcept {
        std::size_t seed = 0;
        hash_combine(seed, p.first.first, p.first.second, p.second);
        return seed;
    }
};

/**
 * @brief Generates a hash value for a double using std::bit_cast for efficiency.
 */
constexpr std::size_t hash_double(double value, std::size_t seed) noexcept {
    // Convert double to uint64_t using bit_cast and hash it with XXH3
    std::uint64_t bit_rep = std::bit_cast<std::uint64_t>(value);
    return XXH3_64bits_withSeed(&bit_rep, sizeof(bit_rep), seed);
}

// Specialize std::hash for std::pair<int, int>
namespace std {
template <>
struct hash<std::pair<int, int>> {
    constexpr std::size_t operator()(const std::pair<int, int> &pair) const noexcept {
        std::size_t h1 = XXH3_64bits_withSeed(&pair.first, sizeof(int), 0);
        std::size_t h2 = XXH3_64bits_withSeed(&pair.second, sizeof(int), h1);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};
} // namespace std

struct VectorHash {
    std::size_t operator()(const std::vector<int> &vec) const {
        // Directly hash the data in the vector using xxh3
        return XXH3_64bits(vec.data(), vec.size() * sizeof(int));
    }
};

struct VectorIntHash {
    std::size_t operator()(const std::vector<int> &vec) const {
        return XXH64(vec.data(), vec.size() * sizeof(int), 0); // Seed = 0
    }
};

// Hash and comparison class for tbb::concurrent_hash_map
struct VectorIntHashCompare {
    using KeyType = std::vector<int>;

    std::size_t hash(const KeyType& key) const {
        return XXH64(key.data(), key.size() * sizeof(int), 0); // Seed = 0
    }

    bool equal(const KeyType& lhs, const KeyType& rhs) const {
        return lhs == rhs;
    }
};