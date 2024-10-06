#pragma once

#include <bit> // for std::rotr (C++20)
#include <cstdint>
#include <cstring> // for std::memcpy
#include <functional>
#include <utility> // for std::pair

/**
 * @brief Combines hash values for multiple values using C++20, without recursion.
 */
template <typename T, typename... Rest>
constexpr void hash_combine(std::size_t &seed, const T &value, const Rest &...rest) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    // Process the remaining values iteratively rather than recursively
    (..., (seed ^= std::hash<Rest>{}(rest) + 0x9e3779b9 + (seed << 6) + (seed >> 2)));
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
    // Direct bit manipulation and XOR with index
    std::uint64_t bit_rep = std::bit_cast<std::uint64_t>(value);
    return bit_rep ^ (index * 0x9e3779b9);
}

// Specialize std::hash for std::pair<int, int>
namespace std {
template <>
struct hash<std::pair<int, int>> {
    constexpr std::size_t operator()(const std::pair<int, int> &pair) const noexcept {
        std::size_t h1 = std::hash<int>{}(pair.first);
        std::size_t h2 = std::hash<int>{}(pair.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};
} // namespace std
