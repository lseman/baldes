#include <array>
#include <bitset>
#include <cstdint>
#include <numeric> // for std::accumulate
#include <stdexcept>

template <size_t N_BITS>
class Bitset {
private:
    static constexpr size_t NUM_WORDS = (N_BITS + 63) / 64;

    // Helper functions for bit index calculations
    static constexpr size_t wordIndex(size_t bit) { return bit / 64; }

    static constexpr size_t bitOffset(size_t bit) { return bit % 64; }

public:
    std::array<uint64_t, NUM_WORDS> bits_{};

    // Bit proxy class for enabling read/write access with `[]`
    class BitReference {
    private:
        uint64_t &word_;
        uint64_t  mask_;

    public:
        BitReference(uint64_t &word, uint64_t mask) : word_(word), mask_(mask) {}

        // Assignment operator
        BitReference &operator=(bool value) {
            if (value)
                word_ |= mask_;
            else
                word_ &= ~mask_;
            return *this;
        }

        // Conversion to bool for reading the bit
        operator bool() const { return (word_ & mask_) != 0; }

        // Assignment from another BitReference
        BitReference &operator=(const BitReference &other) { return *this = bool(other); }
    };

    // Subscript operator for read/write access
    BitReference operator[](size_t bit) {
        return BitReference(bits_[wordIndex(bit)], 1ULL << bitOffset(bit));
    }

    // Const subscript operator for read-only access
    bool operator[](size_t bit) const {
        return (bits_[wordIndex(bit)] >> bitOffset(bit)) & 1ULL;
    }

    // Other member functions from before...

    void set(size_t bit) {
        bits_[wordIndex(bit)] |= (1ULL << bitOffset(bit));
    }

    void clear(size_t bit) {
        bits_[wordIndex(bit)] &= ~(1ULL << bitOffset(bit));
    }

    bool test(size_t bit) const {
        return (bits_[wordIndex(bit)] >> bitOffset(bit)) & 1ULL;
    }

    void toggle(size_t bit) {
        bits_[wordIndex(bit)] ^= (1ULL << bitOffset(bit));
    }

    size_t count() const {
        return std::accumulate(bits_.begin(), bits_.end(), 0ULL,
                               [](size_t total, uint64_t word) { return total + __builtin_popcountll(word); });
    }

    // define .none() and .all() functions
    bool none() const {
        // Return true if no word contains a set bit (i.e., all are zero)
        return std::none_of(bits_.begin(), bits_.end(), [](uint64_t word) { return word != 0; });
    }

    bool all() const {
        // Return true if all words are fully set (i.e., all are UINT64_MAX)
        return std::all_of(bits_.begin(), bits_.end(), [](uint64_t word) { return word == UINT64_MAX; });
    }

    // Bitwise AND operator
    Bitset operator&(const Bitset &other) const {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) { result.bits_[i] = bits_[i] & other.bits_[i]; }
        return result;
    }

    // Bitwise OR operator
    Bitset operator|(const Bitset &other) const {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) { result.bits_[i] = bits_[i] | other.bits_[i]; }
        return result;
    }

    // Bitwise XOR operator
    Bitset operator^(const Bitset &other) const {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) { result.bits_[i] = bits_[i] ^ other.bits_[i]; }
        return result;
    }

    // Bitwise NOT operator
    Bitset operator~() const {
        Bitset result;
        for (size_t i = 0; i < NUM_WORDS; ++i) { result.bits_[i] = ~bits_[i]; }
        return result;
    }

    // Logical operators (`&=`, `|=`, `^=`)
    Bitset &operator&=(const Bitset &other) {
        for (size_t i = 0; i < NUM_WORDS; ++i) { bits_[i] &= other.bits_[i]; }
        return *this;
    }

    Bitset &operator|=(const Bitset &other) {
        for (size_t i = 0; i < NUM_WORDS; ++i) { bits_[i] |= other.bits_[i]; }
        return *this;
    }

    Bitset &operator^=(const Bitset &other) {
        for (size_t i = 0; i < NUM_WORDS; ++i) { bits_[i] ^= other.bits_[i]; }
        return *this;
    }

    // Constructor to initialize from an integer (like 0)
    Bitset(uint64_t value = 0) {
        bits_.fill(0); // Initialize all bits to 0
        if (value != 0) {
            bits_[0] = value; // If non-zero, set only the first word
        }
    }

    // Copy assignment from an integer to clear or partially set the bitset
    Bitset &operator=(uint64_t value) {
        bits_.fill(0);
        if (value != 0) { bits_[0] = value; }
        return *this;
    }

    bool operator==(const Bitset &other) const { return bits_ == other.bits_; }

    // Inequality operator
    bool operator!=(const Bitset &other) const { return !(*this == other); }
};

namespace std {
template <size_t N_BITS>
struct hash<Bitset<N_BITS>> {
    size_t operator()(const Bitset<N_BITS> &bitset) const noexcept {
        // Hash the underlying array of uint64_t using XXH3_64bits
        return XXH3_64bits(bitset.bits_.data(), bitset.bits_.size() * sizeof(uint64_t));
    }
};
} // namespace std