#pragma once

#include <bit> // for std::rotr (C++20)
#include <cstdint>
#include <cstring> // for std::memcpy
#include <functional>
#include <utility> // for std::pair

#include <xxhash.h>  // Include XXHash header


/**
 * @brief Combines hash values for multiple values using C++20, without recursion.
 */
template <typename T, typename... Rest>
constexpr void hash_combine(std::size_t &seed, const T &value, const Rest &...rest) noexcept {
    // Use XXHash instead of std::hash for faster hashing
    seed ^= XXH64(&value, sizeof(T), seed); 
    (..., (seed ^= XXH64(&rest, sizeof(Rest), seed)));
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
constexpr std::size_t hash_double(double value, std::size_t index) noexcept {
    // Convert double to uint64_t using bit_cast and hash it with XXHash
    std::uint64_t bit_rep = std::bit_cast<std::uint64_t>(value);
    return XXH64(&bit_rep, sizeof(bit_rep), index);
}


// Specialize std::hash for std::pair<int, int>
namespace std {
template <>
struct hash<std::pair<int, int>> {
    constexpr std::size_t operator()(const std::pair<int, int> &pair) const noexcept {
        std::size_t h1 = XXH64(&pair.first, sizeof(int), 0);
        std::size_t h2 = XXH64(&pair.second, sizeof(int), h1);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};
}  // namespace std