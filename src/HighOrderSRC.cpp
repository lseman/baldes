
#include "cuts/HighOrderSRC.h"

#include "Bitset.h"

#include "../third_party/small_vector.hpp"

void HighDimCutsGenerator::generateOptimalMultiplier() {
    map_rank1_multiplier[max_heuristic_initial_seed_set_size_row_rank1c + 1].resize(10);
    ankerl::unordered_dense::map<int, Bitset<8>> support;

    for (int i = 1; i <= max_heuristic_initial_seed_set_size_row_rank1c; ++i) {
        auto &it  = map_rank1_multiplier[i];
        auto &sup = support[i];
        it.resize(7);

        // Plan 0
        if (i % 2 != 0) {
            get<0>(it[0]).assign(i, 1);
            get<1>(it[0]) = 2;
            get<2>(it[0]) = i / 2;
            sup.set(0);
        }
        // Plan 1
        if ((i - 2) % 3 == 0 && i >= 5) {
            get<0>(it[1]).assign(i, 1);
            get<1>(it[1]) = 3;
            get<2>(it[1]) = i / 3;
            sup.set(1);
        }

        // Plans for higher ranks, requires i >= 5
        if (i >= 5) {
            auto &tmp = get<0>(it[2]);
            tmp.assign(i, 1);
            tmp[0] = tmp[1] = i - 3;
            tmp[2]          = 2;
            get<1>(it[2])   = i - 2;
            get<2>(it[2])   = 2;
            sup.set(2);

            auto &tmp2 = get<0>(it[3]);
            tmp2.assign(i, 1);
            tmp2[0] = tmp2[1] = i - 2;
            tmp2[2] = tmp2[3] = 2;
            get<1>(it[3])     = i - 1;
            get<2>(it[3])     = 2;
            sup.set(3);

            auto &tmp3 = get<0>(it[4]);
            tmp3.assign(i, 1);
            tmp3[0]       = i - 3;
            tmp3[1]       = 2;
            get<1>(it[4]) = i - 1;
            get<2>(it[4]) = 1;
            sup.set(4);

            auto &tmp4 = get<0>(it[6]);
            tmp4.assign(i, 1);
            tmp4[0] = i - 2;
            tmp4[1] = tmp4[2] = 2;
            get<1>(it[6])     = i;
            get<2>(it[6])     = 1;
            sup.set(6);
        }

        // Plan 5, requires i >= 4
        if (i >= 4) {
            auto &tmp5 = get<0>(it[5]);
            tmp5.assign(i, 1);
            tmp5[0]       = i - 2;
            get<1>(it[5]) = i - 1;
            get<2>(it[5]) = 1;
            sup.set(5);
        }
    }

    // Populate dominance map
    for (int i = 1; i <= max_heuristic_initial_seed_set_size_row_rank1c; ++i) {
        for (int j = 1; j <= max_heuristic_initial_seed_set_size_row_rank1c; ++j) {
            if (i == j || support[i].count() < support[j].count()) continue;
            if (i > j && map_rank1_multiplier_dominance[j].contains(i)) continue;
            if ((support[i] & support[j]) == support[j]) { map_rank1_multiplier_dominance[i].emplace(j); }
        }
    }

    // Generate combinations
    record_map_rank1_combinations.resize(max_heuristic_initial_seed_set_size_row_rank1c + 1,
                                         std::vector<std::vector<std::vector<int>>>(10));

    for (int i = 1; i <= max_heuristic_initial_seed_set_size_row_rank1c; ++i) {
        for (int j = 0; j < 7; ++j) {
            if (get<1>(map_rank1_multiplier[i][j]) == 0) continue;

            ankerl::unordered_dense::map<int, int> count_map;
            for (int val : get<0>(map_rank1_multiplier[i][j])) { ++count_map[val]; }

            std::vector<int>              result;
            std::vector<std::vector<int>> results;
            generatePermutations(count_map, result, results, i);
            record_map_rank1_combinations[i][j] = std::move(results);
        }
    }
}
void HighDimCutsGenerator::printCuts() {
    for (auto &cut : cuts) {
        fmt::print("Cut: ");
        // print info_r1c
        auto par = cut.info_r1c;
        fmt::print("Info_r1c: ");
        for (int i = 0; i < par.first.size(); ++i) { fmt::print("{} ", par.first[i]); }
        fmt::print("Plan: {}\n", par.second);
        fmt::print("RHS: {}\n", cut.rhs);
        fmt::print("Arc_mem: ");
        for (auto &arc : cut.arc_mem) { fmt::print(" {}", arc); }
        fmt::print("\n");
    }
}
void HighDimCutsGenerator::printMultiplierMap() {
    for (const auto &rank : map_rank1_multiplier) {
        std::cout << "Rank: " << rank.first << "\n";
        for (size_t i = 0; i < rank.second.size(); ++i) {
            const auto &plan         = rank.second[i];
            const auto &coefficients = std::get<0>(plan);
            int         denominator  = std::get<1>(plan);
            int         rhs          = std::get<2>(plan);
            if (!coefficients.empty()) {
                std::cout << "  Plan " << i << ": Coefficients = [ ";
                for (int coeff : coefficients) std::cout << coeff << " ";
                std::cout << "], Denominator = " << denominator << ", RHS = " << rhs << "\n";
            }
        }
    }
}
void HighDimCutsGenerator::generatePermutations(ankerl::unordered_dense::map<int, int> &count_map,
                                                std::vector<int> &result, std::vector<std::vector<int>> &results,
                                                int remaining) {
    if (remaining == 0) { // Base case: all elements used
        results.push_back(result);
        return;
    }

    for (auto &[key, count] : count_map) {
        if (count > 0) { // Only proceed if there are remaining elements
            result.push_back(key);
            --count;

            generatePermutations(count_map, result, results, remaining - 1);

            ++count;           // Revert the count
            result.pop_back(); // Backtrack
        }
    }
}
