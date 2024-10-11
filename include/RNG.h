#ifndef XOROSHIRO128PLUS_H
#define XOROSHIRO128PLUS_H

#include <cstdint> // for std::uint64_t

class Xoroshiro128Plus {
    std::uint64_t state0_;
    std::uint64_t state1_;

    // Rotate left helper function
    static inline std::uint64_t rotl(const std::uint64_t x, int k) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_rotateleft64(x, k);
#else
        return (x << k) | (x >> (64 - k));
#endif
    }

public:
    using result_type = std::uint64_t;

    // Constructor that initializes the state with a seed.
    Xoroshiro128Plus(std::uint64_t seed = 42) {
        state0_ = splitmix64(seed);
        state1_ = splitmix64(state0_);
    }

    // Return the minimum value this RNG can generate.
    static constexpr result_type min() { return 0; }

    // Return the maximum value this RNG can generate.
    static constexpr result_type max() { return ~std::uint64_t(0); }

    // Generates a new random number using Xoroshiro128+.
    inline result_type operator()() {
        const std::uint64_t s0 = state0_;
        std::uint64_t       s1 = state1_;

        const std::uint64_t result = s0 + s1;

        s1 ^= s0;
        state0_ = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        state1_ = rotl(s1, 36);                   // c

        return result;
    }

private:
    // SplitMix64 function used for seeding
    static inline std::uint64_t splitmix64(std::uint64_t &x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15);
        z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z               = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
};

#endif
