#pragma once

// Optimized Permutation structure
struct Permutation {
    std::vector<int> num;
    int den;

    Permutation(const std::vector<int>& n, int d) : num(n), den(d) {}
};

class PermutationRegistry {
   private:
    static constexpr int SIZE4_COUNT = 4;
    static constexpr std::array<std::array<int, 4>, 4> SIZE4_BASE = {
        {{2, 1, 1, 1}, {1, 2, 1, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}}};

    static constexpr int SIZE5_COUNT = 85;
    static constexpr std::array<std::array<int, 5>, 85> SIZE5_BASE = {
        {// Group 1: {2, 2, 1, 1, 1} / 4 - 10 permutations
         {2, 2, 1, 1, 1},
         {2, 1, 2, 1, 1},
         {2, 1, 1, 2, 1},
         {2, 1, 1, 1, 2},
         {1, 2, 2, 1, 1},
         {1, 2, 1, 2, 1},
         {1, 2, 1, 1, 2},
         {1, 1, 2, 2, 1},
         {1, 1, 2, 1, 2},
         {1, 1, 1, 2, 2},

         // Group 2: {3, 1, 1, 1, 1} / 4 - 5 permutations
         {3, 1, 1, 1, 1},
         {1, 3, 1, 1, 1},
         {1, 1, 3, 1, 1},
         {1, 1, 1, 3, 1},
         {1, 1, 1, 1, 3},

         // Group 3: {3, 2, 2, 1, 1} / 5 - 30 permutations
         {3, 2, 2, 1, 1},
         {3, 2, 1, 2, 1},
         {3, 2, 1, 1, 2},
         {3, 1, 2, 2, 1},
         {3, 1, 2, 1, 2},
         {3, 1, 1, 2, 2},
         {2, 3, 2, 1, 1},
         {2, 3, 1, 2, 1},
         {2, 3, 1, 1, 2},
         {2, 2, 3, 1, 1},
         {2, 2, 1, 3, 1},
         {2, 2, 1, 1, 3},
         {2, 1, 3, 2, 1},
         {2, 1, 3, 1, 2},
         {2, 1, 2, 3, 1},
         {2, 1, 2, 1, 3},
         {2, 1, 1, 3, 2},
         {2, 1, 1, 2, 3},
         {1, 3, 2, 2, 1},
         {1, 3, 2, 1, 2},
         {1, 3, 1, 2, 2},
         {1, 2, 3, 2, 1},
         {1, 2, 3, 1, 2},
         {1, 2, 2, 3, 1},
         {1, 2, 2, 1, 3},
         {1, 2, 1, 3, 2},
         {1, 2, 1, 2, 3},
         {1, 1, 3, 2, 2},
         {1, 1, 2, 3, 2},
         {1, 1, 2, 2, 3},

         // Group 4: {2, 2, 1, 1, 1} / 3 - 10 permutations
         {2, 2, 1, 1, 1},
         {2, 1, 2, 1, 1},
         {2, 1, 1, 2, 1},
         {2, 1, 1, 1, 2},
         {1, 2, 2, 1, 1},
         {1, 2, 1, 2, 1},
         {1, 2, 1, 1, 2},
         {1, 1, 2, 2, 1},
         {1, 1, 2, 1, 2},
         {1, 1, 1, 2, 2},

         // Group 5: {3, 3, 2, 2, 1} / 4 - 30 permutations
         {3, 3, 2, 2, 1},
         {3, 3, 2, 1, 2},
         {3, 3, 1, 2, 2},
         {3, 2, 3, 2, 1},
         {3, 2, 3, 1, 2},
         {3, 2, 2, 3, 1},
         {3, 2, 2, 1, 3},
         {3, 2, 1, 3, 2},
         {3, 2, 1, 2, 3},
         {3, 1, 3, 2, 2},
         {3, 1, 2, 3, 2},
         {3, 1, 2, 2, 3},
         {2, 3, 3, 2, 1},
         {2, 3, 3, 1, 2},
         {2, 3, 2, 3, 1},
         {2, 3, 2, 1, 3},
         {2, 3, 1, 3, 2},
         {2, 3, 1, 2, 3},
         {2, 2, 3, 3, 1},
         {2, 2, 3, 1, 3},
         {2, 2, 1, 3, 3},
         {2, 1, 3, 3, 2},
         {2, 1, 3, 2, 3},
         {2, 1, 2, 3, 3},
         {1, 3, 3, 2, 2},
         {1, 3, 2, 3, 2},
         {1, 3, 2, 2, 3},
         {1, 2, 3, 3, 2},
         {1, 2, 3, 2, 3},
         {1, 2, 2, 3, 3}}};

    static constexpr std::array<int, 85> SIZE5_DENOMINATORS = {
        {// Group 1: 10 elements
         4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         // Group 2: 5 elements
         4, 4, 4, 4, 4,
         // Group 3: 30 elements
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 5, 5, 5,
         // Group 4: 10 elements
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         // Group 5: 30 elements
         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4}};

    // Cache for constructed permutations
    alignas(64) inline static std::vector<Permutation> size4_cache;
    alignas(64) inline static std::vector<Permutation> size5_cache;
    inline static bool initialized = false;

   public:
    static void initialize() {
        if (!initialized) {
            // Pre-allocate space
            size4_cache.reserve(SIZE4_COUNT);
            size5_cache.reserve(SIZE5_COUNT);

            // Initialize size 4 cache
            for (int i = 0; i < SIZE4_COUNT; ++i) {
                std::vector<int> nums(SIZE4_BASE[i].begin(),
                                      SIZE4_BASE[i].end());
                size4_cache.emplace_back(nums, 3);
            }

            // Initialize size 5 cache
            for (int i = 0; i < SIZE5_COUNT; ++i) {
                std::vector<int> nums(SIZE5_BASE[i].begin(),
                                      SIZE5_BASE[i].end());
                size5_cache.emplace_back(nums, SIZE5_DENOMINATORS[i]);
            }

            initialized = true;
        }
    }

    static const std::vector<Permutation>& getCachedPermutations(int size) {
        initialize();  // Ensure initialization
        if (size == 4) {
            return size4_cache;
        } else if (size == 5) {
            return size5_cache;
        }
        static const std::vector<Permutation> empty;
        return empty;
    }
};

// Replace original functions
inline std::vector<Permutation> getPermutationsForSize5() {
    return PermutationRegistry::getCachedPermutations(5);
}

inline std::vector<Permutation> getPermutationsForSize4() {
    return PermutationRegistry::getCachedPermutations(4);
}
