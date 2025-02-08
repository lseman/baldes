/*
 * @file Bitset.h
 * @brief This file contains the definition of the Bitset class.
 *
 * This file contains the definition of the Bitset class, which represents a
 * fixed-size bitset. The Bitset class provides methods to set, clear, test,
 * toggle, count, check if all bits are set, check if no bits are set, perform
 * bitwise AND, OR, XOR, and NOT operations, and compare bitsets. It also
 * provides methods to convert to an integer, hash the bitset, and stream the
 * bitset. The Bitset class is implemented using a std::array of uint64_t to
 * store the bits.
 *
 */
#pragma once

#include <immintrin.h>  // For SIMD intrinsics
#include <xxhash.h>     // For fast hashing

#include <array>
#include <cstdint>
#include <execution>  // For parallel algorithms
#include <numeric>

template <size_t N_BITS>
class Bitset {
   private:
    static constexpr size_t NUM_WORDS = (N_BITS + 63) / 64;

    static constexpr size_t wordIndex(size_t bit) { return bit / 64; }
    static constexpr size_t bitOffset(size_t bit) { return bit % 64; }

   public:
    alignas(32) std::array<uint64_t, NUM_WORDS> bits_{};

    constexpr Bitset(uint64_t value = 0) {
        bits_.fill(0);
        if (value != 0) {
            bits_[0] = value;
        }
    }

    constexpr void set(size_t bit) noexcept {
        bits_[wordIndex(bit)] |= (1ULL << bitOffset(bit));
    }
    constexpr void clear(size_t bit) noexcept {
        bits_[wordIndex(bit)] &= ~(1ULL << bitOffset(bit));
    }
    constexpr bool test(size_t bit) const noexcept {
        return (bits_[wordIndex(bit)] >> bitOffset(bit)) & 1ULL;
    }
    constexpr void toggle(size_t bit) noexcept {
        bits_[wordIndex(bit)] ^= (1ULL << bitOffset(bit));
    }
    constexpr void reset() noexcept { bits_.fill(0); }

    // Const subscript operator for read-only access
    constexpr bool operator[](size_t bit) const noexcept {
        return (bits_[wordIndex(bit)] >> bitOffset(bit)) & 1ULL;
    }

    constexpr bool none() const noexcept {
        for (const auto &word : bits_) {
            if (word != 0) return false;
        }
        return true;
    }

    constexpr bool all() const noexcept {
        for (const auto &word : bits_) {
            if (word != UINT64_MAX) return false;
        }
        return true;
    }

    size_t count() const noexcept {
#ifdef __AVX2__
        size_t total = 0;
        for (const auto &word : bits_) {
            total += __builtin_popcountll(word);
        }
        return total;
#else
        return std::accumulate(bits_.begin(), bits_.end(), 0ULL,
                               [](size_t total, uint64_t word) {
                                   return total + __builtin_popcountll(word);
                               });
#endif
    }

    Bitset operator&(const Bitset &other) const noexcept {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            result.bits_[i] = bits_[i] & other.bits_[i];
        }
        return result;
    }

    Bitset operator|(const Bitset &other) const noexcept {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            result.bits_[i] = bits_[i] | other.bits_[i];
        }
        return result;
    }

    Bitset operator^(const Bitset &other) const noexcept {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            result.bits_[i] = bits_[i] ^ other.bits_[i];
        }
        return result;
    }

    Bitset operator~() const noexcept {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            result.bits_[i] = ~bits_[i];
        }
        return result;
    }

    Bitset &operator&=(const Bitset &other) noexcept {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            bits_[i] &= other.bits_[i];
        }
        return *this;
    }

    Bitset &operator|=(const Bitset &other) noexcept {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            bits_[i] |= other.bits_[i];
        }
        return *this;
    }

    Bitset &operator^=(const Bitset &other) noexcept {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            bits_[i] ^= other.bits_[i];
        }
        return *this;
    }

    bool operator==(const Bitset &other) const noexcept {
        return bits_ == other.bits_;
    }
    bool operator!=(const Bitset &other) const noexcept {
        return !(*this == other);
    }

    friend std::ostream &operator<<(std::ostream &os, const Bitset &bitset) {
        for (size_t i = 0; i < N_BITS; ++i) {
            os << bitset.test(i);
        }
        return os;
    }

    ~Bitset() noexcept { bits_.fill(0); }
};

namespace std {
template <size_t N_BITS>
struct hash<Bitset<N_BITS>> {
    size_t operator()(const Bitset<N_BITS> &bitset) const noexcept {
        return XXH3_64bits(bitset.bits_.data(),
                           bitset.bits_.size() * sizeof(uint64_t));
    }
};
}  // namespace std
