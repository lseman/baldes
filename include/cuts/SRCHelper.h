/*
* @file HighDimCutsGenerator.h
* @brief High-dimensional cuts generator for the VRP
*
* This class is responsible for generating high-dimensional cuts for the VRP.
* The class is based on RouteOpt: https://github.com/Zhengzhong-You/RouteOpt/
*
*/

#pragma once

#include "Path.h"
#include "SparseMatrix.h"
#include <bitset>
#include <iostream>
#include <tuple>
#include <vector>

#include <algorithm>
#include <bitset>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>


#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

constexpr int INITIAL_RANK_1_MULTI_LABEL_POOL_SIZE              = 100;
using yzzLong                                                   = std::bitset<N_SIZE>;
using cutLong                                                   = yzzLong;
constexpr double tolerance                                      = 1e-6;
constexpr int    max_row_rank1                                  = 5;
constexpr int    max_heuristic_initial_seed_set_size_row_rank1c = 5 + 1;

constexpr int    max_num_r1c_per_round = 10;
constexpr double max_cut_mem_factor    = 0.35;
constexpr double cut_vio_factor        = 0.1;

#define INITIAL_IDX_R1C (-1)

struct VectorHash {
    std::size_t operator()(const std::vector<int> &vec) const {
        // Directly hash the data in the vector using xxh3
        return XXH3_64bits(vec.data(), vec.size() * sizeof(int));
    }
};

struct other_ {
    int              beg{};
    int              tor{};
    std::vector<int> left_c{};
    std::vector<int> mem_c{};

    other_(int beg, int tor, std::vector<int> left_c, std::vector<int> mem_c)
        : beg(beg), tor(tor), left_c(std::move(left_c)), mem_c(std::move(mem_c)) {}

    other_() = default;
};

struct R1c {
    std::pair<std::vector<int>, int>              info_r1c{}; // cut and plan
    int                                           idx_r1c{INITIAL_IDX_R1C};
    int                                           rhs{}; // get<2>map[cut.size()][plan_idx]
    std::vector<std::pair<std::vector<int>, int>> arc_mem{};
};

class HighDimCutsGenerator {
public:
    HighDimCutsGenerator(int dim, int maxRowRank, double tolerance)
        : dim(dim), max_row_rank1(maxRowRank), TOLERANCE(tolerance), num_label(0) {
        generateOptimalMultiplier();
    }

    void generateOptimalMultiplier();
    void printCuts();

    std::vector<R1c> &getCuts() { return cuts; }

    ankerl::unordered_dense::map<int, std::vector<std::tuple<std::vector<int>, int, int>>> map_rank1_multiplier;
    ankerl::unordered_dense::map<int, ankerl::unordered_dense::set<int>> map_rank1_multiplier_dominance{};
    std::vector<std::vector<std::vector<std::vector<int>>>>              record_map_rank1_combinations{};

    void printMultiplierMap();

    void generatePermutations(ankerl::unordered_dense::map<int, int> &count_map, std::vector<int> &result,
                              std::vector<std::vector<int>> &results, int remaining);

    void initialize(const SparseMatrix &matrix, const std::vector<Path> &routes, const std::vector<double> &solution) {
        this->matrix   = matrix;
        this->routes   = routes;
        this->solution = solution;
        this->sol      = routes;
        initialSupportvector();
    }

    std::vector<VRPNode> nodes;
    void                 setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    struct Rank1MultiLabel {
        std::vector<int> c;
        std::vector<int> w_no_c;
        int              plan_idx{};
        double           vio{};
        char             search_dir{};

        Rank1MultiLabel(std::vector<int> c, std::vector<int> w_no_c, int plan_idx, double vio, char search_dir)
            : c(std::move(c)), w_no_c(std::move(w_no_c)), plan_idx(plan_idx), vio(vio), search_dir(search_dir) {}

        Rank1MultiLabel() = default;
    };

    std::vector<Path>                                          sol;
    int                                                        dim, max_row_rank1, num_label;
    double                                                     TOLERANCE;
    SparseMatrix                                               matrix;
    std::vector<Path>                                          routes;
    std::vector<double>                                        solution;
    std::vector<ankerl::unordered_dense::map<int, int>>        v_r_map;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> c_N_noC;
    ankerl::unordered_dense::map<std::bitset<N_SIZE>, std::vector<std::pair<std::vector<int>, double>>>
                                 map_cut_plan_vio;
    std::vector<Rank1MultiLabel> rank1_multi_label_pool;
    ankerl::unordered_dense::map<int, std::vector<std::tuple<std::bitset<N_SIZE>, int, double>>>
                     generated_rank1_multi_pool;
    std::vector<R1c> cuts;
    std::vector<R1c> old_cuts;

    ankerl::unordered_dense::map<std::vector<int>, std::vector<std::vector<int>>, VectorHash> rank1_multi_mem_plan_map;
    ankerl::unordered_dense::map<std::vector<int>, ankerl::unordered_dense::set<int>, VectorHash> cut_record;
    std::vector<std::pair<int, double>>                                                           move_vio;
    std::vector<yzzLong> rank1_sep_heur_mem4_vertex;

    void initialSupportvector() {
        cut_record.clear();
        v_r_map.clear();
        c_N_noC.clear();
        map_cut_plan_vio.clear();
        map_cut_plan_vio.reserve(4096);
        generated_rank1_multi_pool.clear();
        rank1_multi_label_pool.clear();
        num_label = 0;
        rank1_multi_mem_plan_map.clear();
        cuts.clear();
        rank1_multi_label_pool.resize(INITIAL_RANK_1_MULTI_LABEL_POOL_SIZE);
    }

    void getHighDimCuts() {
        constructVRMapAndSeedCrazy();
        startSeedCrazy();
        for (int i = 0; i < num_label;) { operationsCrazy(rank1_multi_label_pool[i], i); }
        constructCutsCrazy();
    }

    void processSeedForPlan(int plan_idx, std::vector<Rank1MultiLabel> &local_pool) {
        for (auto &[c, wc] : c_N_noC) {
            double vio, best_vio;
            exactFindBestPermutationForOnePlan(c, plan_idx, vio);
            if (vio < TOLERANCE) continue;

            best_vio                      = vio;
            char                best_oper = 'o';
            int                 add_j = -1, remove_j = -1;
            std::pair<int, int> swap_i_j = {-1, -1};

            // Perform the searches
            addSearchCrazy(plan_idx, c, wc, vio, add_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 'a';
            }

            removeSearchCrazy(plan_idx, c, vio, remove_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 'r';
            }

            swapSearchCrazy(plan_idx, c, wc, vio, swap_i_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 's';
            }

            cutLong          tmp;
            std::vector<int> new_c, new_w_no_c;

            // Use optimized operations based on `best_oper`
            switch (best_oper) {
            case 'o':
                if (c.size() < 2 || c.size() > max_row_rank1) break;
                tmp = 0;
                for (int i : c) tmp.set(i);
                generated_rank1_multi_pool[static_cast<int>(c.size())].emplace_back(tmp, plan_idx, best_vio);
                break;

            case 'a': // Add operation
                new_c = c;
                new_c.push_back(add_j);
                new_w_no_c.reserve(wc.size() - 1);
                std::copy_if(wc.begin(), wc.end(), std::back_inserter(new_w_no_c),
                             [add_j](int w) { return w != add_j; });
                break;

            case 'r': // Remove operation
                new_c.reserve(c.size() - 1);
                std::copy_if(c.begin(), c.end(), std::back_inserter(new_c),
                             [remove_j](int i) { return i != remove_j; });
                new_w_no_c = wc;
                break;

            case 's': // Swap operation
                new_c      = c;
                new_w_no_c = wc;
                for (int &i : new_c) {
                    if (i == swap_i_j.first) i = swap_i_j.second;
                }
                for (int &i : new_w_no_c) {
                    if (i == swap_i_j.second) i = swap_i_j.first;
                }
                break;
            }

            // Only add to local pool if an operation was performed
            if (best_oper != 'o') {
                local_pool.emplace_back(std::move(new_c), std::move(new_w_no_c), plan_idx, best_vio, best_oper);
            }
        }
    }

    std::mutex pool_mutex; // Mutex to protect shared pool access

    void startSeedCrazy() {
        num_label                     = 0;
        std::vector<int> plan_indices = {0, 1, 2, 3, 4, 5, 6};

        const int                JOBS = plan_indices.size();
        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        auto bulk_sender = stdexec::bulk(stdexec::just(), JOBS, [&](std::size_t plan_idx) {
            std::vector<Rank1MultiLabel> local_pool; // Local pool to collect results per thread
            processSeedForPlan(plan_idx, local_pool);

            // Lock and transfer data from local pool to shared pool
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (auto &label : local_pool) {
                if (num_label >= rank1_multi_label_pool.size()) {
                    rank1_multi_label_pool.resize(rank1_multi_label_pool.size() * 1.5);
                }
                rank1_multi_label_pool[num_label++] = std::move(label);
            }
        });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
    }

    // Assuming map_cut_plan_vio is declared as follows
    std::mutex map_cut_plan_vio_mutex; // Mutex for protecting map_cut_plan_vio
    void       exactFindBestPermutationForOnePlan(std::vector<int> &cut, const int plan_idx, double &vio) {
        const int cut_size = static_cast<int>(cut.size());

        const auto &plan = map_rank1_multiplier[cut_size][plan_idx];

        // Early exit if the plan denominator is zero
        if (get<1>(plan) == 0) {
            vio = -std::numeric_limits<double>::max();
            return;
        }

        cutLong tmp = 0;
        for (const auto &it : cut) {
            if (it >= 0 && it < v_r_map.size()) tmp.set(it);
        }

        {
            std::lock_guard<std::mutex> lock(map_cut_plan_vio_mutex);
            // Check if cached result exists and is valid
            if (auto it = map_cut_plan_vio.find(tmp);
                it != map_cut_plan_vio.end() && plan_idx < it->second.size() && !it->second[plan_idx].first.empty()) {
                vio = it->second[plan_idx].second;
                return;
            } else {
                map_cut_plan_vio[tmp].resize(7);
            }
        }

        const int    denominator = get<1>(plan);
        const double rhs         = get<2>(plan);
        const auto  &coeffs      = record_map_rank1_combinations[cut_size][plan_idx];

        // Count occurrences in `cut`
        ankerl::unordered_dense::map<int, int> map_r_numbers;
        for (const auto &i : cut) {
            if (i >= 0 && i < v_r_map.size()) {
                for (const auto &[fst, snd] : v_r_map[i]) { map_r_numbers[fst] += snd; }
            }
        }

        std::vector<double> valid_routes;
        valid_routes.reserve(sol.size());
        ankerl::unordered_dense::map<int, int> map_old_new_routes;
        map_old_new_routes.reserve(sol.size());

        for (const auto &pr : map_r_numbers) {
            if (pr.first >= 0 && pr.first < sol.size() && pr.second > 1) {
                valid_routes.push_back(sol[pr.first].frac_x);
                map_old_new_routes[pr.first] = static_cast<int>(valid_routes.size()) - 1;
            }
        }

        std::vector<std::vector<int>> cut_num_times_vis_routes(cut_size);
        for (int i = 0; i < cut_size; ++i) {
            int c = cut[i];
            if (c >= 0 && c < v_r_map.size()) {
                for (const auto &pr : v_r_map[c]) {
                    if (auto it = map_old_new_routes.find(pr.first); it != map_old_new_routes.end()) {
                        cut_num_times_vis_routes[i].insert(cut_num_times_vis_routes[i].end(), pr.second, it->second);
                    }
                }
            }
        }

        std::vector<double> num_times_vis_routes(map_old_new_routes.size());
        double              best_vio = -std::numeric_limits<double>::max();
        int                 best_idx = -1;

        for (size_t cnt = 0; cnt < coeffs.size(); ++cnt) {
            std::fill(num_times_vis_routes.begin(), num_times_vis_routes.end(), 0.0);

            for (int i = 0; i < cut_size; ++i) {
                for (int j : cut_num_times_vis_routes[i]) { num_times_vis_routes[j] += coeffs[cnt][i]; }
            }

            double vio_tmp = -rhs;
            for (size_t i = 0; i < num_times_vis_routes.size(); ++i) {
                num_times_vis_routes[i] =
                    static_cast<int>(num_times_vis_routes[i] / denominator + tolerance) * valid_routes[i];
                vio_tmp += num_times_vis_routes[i];
            }

            if (vio_tmp > best_vio) {
                best_vio = vio_tmp;
                best_idx = static_cast<int>(cnt);
            }
        }

        vio = best_vio;

        std::vector<std::pair<int, int>> cut_coeff(cut_size);
        for (int i = 0; i < cut_size; ++i) {
            if (best_idx >= 0 && best_idx < coeffs.size() && i < coeffs[best_idx].size()) {
                cut_coeff[i] = {cut[i], coeffs[best_idx][i]};
            }
        }

        pdqsort(cut_coeff.begin(), cut_coeff.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
        std::vector<int> new_cut(cut_size);
        std::transform(cut_coeff.begin(), cut_coeff.end(), new_cut.begin(), [](const auto &a) { return a.first; });

        {
            std::lock_guard<std::mutex> lock(map_cut_plan_vio_mutex);
            if (plan_idx < map_cut_plan_vio[tmp].size()) {
                map_cut_plan_vio[tmp][plan_idx] = {std::move(new_cut), vio};
            }
        }
    }

    void generateR1C1() {
        std::vector<std::pair<double, R1c>> tmp_cuts;

        ankerl::unordered_dense::map<int, int> vis_map;
        for (const auto &r : sol) {
            vis_map.clear();
            for (const auto i : r.route) { ++vis_map[i]; }
            for (auto &[v, times] : vis_map) {
                if (times > 1) {
                    tmp_cuts.emplace_back();
                    tmp_cuts.back().first           = floor(times / 2. + tolerance) * r.frac_x;
                    tmp_cuts.back().second.info_r1c = make_pair(std::vector{v}, 0);
                }
            }
        }
        if (tmp_cuts.empty()) return;

        pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
                [](const std::pair<double, R1c> &a, const std::pair<double, R1c> &b) { return a.first > b.first; });
        std::vector<R1c> pure_cuts(tmp_cuts.size());
        std::transform(tmp_cuts.begin(), tmp_cuts.end(), pure_cuts.begin(),
                       [](const std::pair<double, R1c> &a) { return a.second; });
        chooseCuts(pure_cuts, cuts, max_num_r1c_per_round);
    }

    const int                        max_heuristic_sep_mem4_row_rank1 = 8;
    std::vector<std::vector<double>> cost_mat4_vertex;
    void                             generateSepHeurMem4Vertex() {
        rank1_sep_heur_mem4_vertex.resize(dim);
        std::vector<std::pair<int, double>> cost(dim);

        for (int i = 0; i < dim; ++i) {
            cost[0] = {0, INFINITY};
            for (int j = 1; j < dim - 1; ++j) { cost[j] = {j, cost_mat4_vertex[i][j]}; }

            // Sort by cost
            pdqsort(cost.begin(), cost.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

            cutLong &vst2 = rank1_sep_heur_mem4_vertex[i];
            for (int k = 0; k < max_heuristic_sep_mem4_row_rank1; ++k) { vst2.set(cost[k].first); }
        }
    }

    void constructVRMapAndSeedCrazy() {
        // Initialize and reserve space for `rank1_sep_heur_mem4_vertex` and `v_r_map`
        rank1_sep_heur_mem4_vertex.resize(dim);
        v_r_map.assign(dim, {});
        for (auto &map_entry : v_r_map) { map_entry.reserve(sol.size()); }

        // Populate v_r_map based on `sol`
        for (int r = 0; r < sol.size(); ++r) {
            for (const int i : sol[r].route) {
                if (i > 0 && i < dim - 1) {
                    auto &entry = v_r_map[i];
                    if (entry.find(r) == entry.end())
                        entry[r] = 1;
                    else
                        ++entry[r];
                }
            }
        }

        // Generate `seed_map` with reserve capacity
        ankerl::unordered_dense::map<cutLong, cutLong> seed_map;
        seed_map.reserve(4096);

        for (int i = 1; i < dim - 1; ++i) {
            if (v_r_map[i].empty()) continue;

            cutLong wc;
            for (const auto &[route_index, count] : v_r_map[i]) {
                for (const int v : sol[route_index].route) {
                    if (v > 0 && v < dim - 1 && rank1_sep_heur_mem4_vertex[i].test(v)) { wc.set(v); }
                }
            }

            // Create seed map entries based on `wc`
            for (int r = 0; r < sol.size(); ++r) {
                if (v_r_map[i].find(r) != v_r_map[i].end()) continue;

                cutLong tmp_c;
                for (const int v : sol[r].route) {
                    if (v > 0 && v < dim - 1 && wc.test(v)) { tmp_c.set(v); }
                }
                tmp_c.set(i);

                int c_size = static_cast<int>(tmp_c.count());
                if (c_size >= 4 && c_size <= max_heuristic_initial_seed_set_size_row_rank1c) {
                    if (auto [it, inserted] = seed_map.try_emplace(tmp_c, wc ^ tmp_c); !inserted) {
                        it->second |= wc ^ tmp_c;
                    }
                }
            }
        }

        // Populate `c_N_noC` based on `seed_map`
        c_N_noC.resize(seed_map.size());
        int cnt = 0;
        for (const auto &[fst, snd] : seed_map) {
            auto &[tmp_fst, tmp_snd] = c_N_noC[cnt];
            tmp_fst.reserve(fst.count());
            tmp_snd.reserve(snd.count());

            for (int i = 1; i < dim - 1; ++i) {
                if (fst.test(i)) {
                    tmp_fst.push_back(i);
                } else if (snd.test(i)) {
                    tmp_snd.push_back(i);
                }
            }
            ++cnt;
        }
    }

    void addSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c, double &new_vio,
                        int &add_j) {
        const int   new_c_size = static_cast<int>(c.size()) + 1;
        const auto &plan       = map_rank1_multiplier[new_c_size][plan_idx];

        if (new_c_size > max_row_rank1 || !get<1>(plan)) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        std::vector<int> tmp_c = c;
        tmp_c.push_back(0);

        double best_vio       = -std::numeric_limits<double>::max();
        int    best_candidate = -1;

        for (const int cplus : w_no_c) {
            tmp_c.back() = cplus;
            double vio;
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

            if (vio > best_vio) {
                best_vio       = vio;
                best_candidate = cplus;
            }
        }

        new_vio = best_vio - tolerance;
        add_j   = best_candidate;
    }

    void removeSearchCrazy(int plan_idx, const std::vector<int> &c, double &new_vio, int &remove_j) {
        const int   new_c_size = static_cast<int>(c.size()) - 1;
        const auto &plan       = map_rank1_multiplier[new_c_size][plan_idx];

        // Check if the size is below the minimum or if the plan is invalid
        if (new_c_size < 3 || !get<1>(plan)) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        std::vector<int> tmp_c(new_c_size);
        double           best_vio       = -std::numeric_limits<double>::max();
        int              best_candidate = -1;

        // Try removing each element from `c` and calculate the violation
        for (int i = 0; i < c.size(); ++i) {
            std::copy(c.begin(), c.begin() + i, tmp_c.begin());
            std::copy(c.begin() + i + 1, c.end(), tmp_c.begin() + i);

            double vio;
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
            if (vio > best_vio) {
                best_vio       = vio;
                best_candidate = c[i];
            }
        }

        new_vio  = best_vio + tolerance; // Apply penalty
        remove_j = best_candidate;
    }

    void swapSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c, double &new_vio,
                         std::pair<int, int> &swap_i_j) {
        const int   c_size = static_cast<int>(c.size());
        const auto &plan   = map_rank1_multiplier[c_size][plan_idx];

        // Early exit if constraints are not met
        if ((c_size < 3 || c_size > max_row_rank1) || !get<1>(plan)) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        std::vector<int>    tmp_c     = c; // Temporary vector for swaps
        double              best_vio  = -std::numeric_limits<double>::max();
        std::pair<int, int> best_swap = {-1, -1};

        // Try swapping each element in `c` with each element in `w_no_c` and calculate the violation
        for (int i = 0; i < c.size(); ++i) {
            const int original = tmp_c[i];

            for (int j : w_no_c) {
                if (original == j) continue; // Skip if elements are the same

                tmp_c[i] = j; // Perform the swap
                double vio;
                exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

                if (vio > best_vio) {
                    best_vio  = vio;
                    best_swap = {original, j};
                }
            }

            tmp_c[i] = original; // Restore the original element after each swap
        }

        new_vio  = best_vio;
        swap_i_j = best_swap;
    }

    void operationsCrazy(Rank1MultiLabel &label, int &i) {
        auto &vio      = label.vio;
        auto &new_cij  = label.c;
        auto &w_no_cij = label.w_no_c;
        auto &plan_idx = label.plan_idx;
        auto  dir      = label.search_dir;

        int                                   add_j = -1, remove_j = -1;
        std::pair<int, int>                   swap_i_j;
        std::array<std::pair<int, double>, 4> move_vio = {{{0, vio},
                                                           {1, -std::numeric_limits<double>::max()},
                                                           {2, -std::numeric_limits<double>::max()},
                                                           {3, -std::numeric_limits<double>::max()}}};

        double new_vio;
        // Check conditions and perform the respective searches
        if (dir == 'a' || dir == 's') {
            addSearchCrazy(plan_idx, new_cij, w_no_cij, new_vio, add_j);
            move_vio[1].second = new_vio;
        } else if (dir == 'r' || dir == 's') {
            removeSearchCrazy(plan_idx, new_cij, new_vio, remove_j);
            move_vio[2].second = new_vio;
        } else if (dir == 'a' || dir == 'r') {
            swapSearchCrazy(plan_idx, new_cij, w_no_cij, new_vio, swap_i_j);
            move_vio[3].second = new_vio;
        }

        // Find the move with the highest violation score without sorting
        auto   best_move_it  = std::max_element(move_vio.begin(), move_vio.end(),
                                                [](const auto &a, const auto &b) { return a.second < b.second; });
        int    best_move     = best_move_it->first;
        double best_move_vio = best_move_it->second;

        cutLong tmp;
        switch (best_move) {
        case 0: // No operation (default best move)
            tmp = 0;
            for (int j : new_cij) { tmp.set(j); }
            generated_rank1_multi_pool[static_cast<int>(new_cij.size())].emplace_back(tmp, plan_idx, best_move_vio);
            ++i;
            break;
        case 1: { // Add operation
            auto pos = std::find(w_no_cij.begin(), w_no_cij.end(), add_j);
            if (pos != w_no_cij.end()) {
                new_cij.push_back(add_j);
                w_no_cij.erase(pos);
            }
            break;
        }
        case 2: { // Remove operation
            auto pos = std::find(new_cij.begin(), new_cij.end(), remove_j);
            if (pos != new_cij.end()) { new_cij.erase(pos); }
            break;
        }
        case 3: { // Swap operation
            auto pos = std::find(new_cij.begin(), new_cij.end(), swap_i_j.first);
            if (pos != new_cij.end()) {
                *pos       = swap_i_j.second;
                auto w_pos = std::find(w_no_cij.begin(), w_no_cij.end(), swap_i_j.second);
                if (w_pos != w_no_cij.end()) { w_no_cij.erase(w_pos); }
            }
            break;
        }
        default: throw std::runtime_error("Invalid best move");
        }
        vio = best_move_vio;
    }

    void chooseCuts(const std::vector<R1c> &tmp_cuts, std::vector<R1c> &chosen_cuts, int numCuts) {
        numCuts = std::min(numCuts, static_cast<int>(tmp_cuts.size()));
        if (numCuts == 0) return;

        for (const auto &cut : tmp_cuts) {
            const auto &fst = cut.info_r1c.first;
            const auto &snd = cut.info_r1c.second;

            // Avoid repeated cuts by checking cut_record
            if (cut_record[fst].count(snd) > 0) continue;
            cut_record[fst].insert(snd);

            int         size  = static_cast<int>(fst.size());
            const auto &coeff = get<0>(map_rank1_multiplier[size][snd]);

            // Organize elements of `fst` by coefficients into `tmp_cut`
            std::vector<std::vector<int>> tmp_cut(coeff[0] + 1);
            for (int i = 0; i < size; ++i) { tmp_cut[coeff[i]].push_back(fst[i]); }

            // Sort each group if it has more than one element
            for (auto &group : tmp_cut) {
                if (group.size() > 1) pdqsort(group.begin(), group.end());
            }

            // Flatten sorted `tmp_cut` into `new_cut`
            std::vector<int> new_cut;
            new_cut.reserve(size);
            for (int i = coeff[0]; i >= 0; --i) { new_cut.insert(new_cut.end(), tmp_cut[i].begin(), tmp_cut[i].end()); }

            // Add the new cut to chosen cuts
            chosen_cuts.emplace_back();
            chosen_cuts.back().info_r1c = std::make_pair(new_cut, snd);

            // Stop if required number of cuts is reached
            if (--numCuts == 0) break;
        }
    }

    void constructCutsCrazy() {
        for (auto &pool : generated_rank1_multi_pool) {
            // Sort cuts within each pool in descending order of violation score
            pdqsort(pool.second.begin(), pool.second.end(),
                    [](const auto &a, const auto &b) { return std::get<2>(a) > std::get<2>(b); });
        }

        for (auto &pool : generated_rank1_multi_pool) {
            ankerl::unordered_dense::set<cutLong> cut_set;
            ankerl::unordered_dense::set<int>     p_set;
            const double                          vio_threshold = std::get<2>(pool.second[0]) * cut_vio_factor;
            std::vector<R1c>                      tmp_cuts;

            // Filter and select cuts based on violation threshold
            for (const auto &cut : pool.second) {
                if (std::get<2>(cut) < vio_threshold) break;

                const auto &key      = std::get<0>(cut);
                int         plan_idx = std::get<1>(cut);

                if (cut_set.contains(key) && p_set.contains(plan_idx)) continue;

                const auto &cut_plan = map_cut_plan_vio.at(key).at(plan_idx);
                tmp_cuts.emplace_back(R1c{std::make_pair(cut_plan.first, plan_idx)});
                cut_set.insert(key);
                p_set.insert(plan_idx);
            }

            // Select the best cuts from tmp_cuts based on max_num_r1c_per_round
            chooseCuts(tmp_cuts, cuts, max_num_r1c_per_round);
        }
    }

    void findPlanForRank1Multi(const std::vector<int> &vis, const int denominator, cutLong &mem,
                               std::vector<ankerl::unordered_dense::set<int>> &segment,
                               std::vector<std::vector<int>>                  &plan) {
        int sum = std::accumulate(vis.begin(), vis.end(), 0);
        int mod = sum % denominator;

        // Prepare key for rank1_multi_mem_plan_map
        std::vector<int> key = vis;
        key.push_back(mod);

        auto &other2 = rank1_multi_mem_plan_map[key];
        if (other2.empty()) {
            // Initialize `other` list with the first element
            std::list<other_> other;
            other.emplace_back(0, mod, vis, std::vector<int>{});

            // Expand `other` list to cover necessary permutations
            for (auto it = other.begin(); it != other.end(); ++it) {
                auto &o   = *it;
                int   cnt = 0;
                int   tor = o.tor;
                int   beg = o.beg;

                for (int j = 0; j < o.left_c.size(); ++j) {
                    cnt += o.left_c[j];
                    cnt %= denominator;

                    if (cnt > 0) {
                        if (cnt <= tor && (o.left_c.begin() + j + 1) < o.left_c.end()) {
                            other.emplace_back(beg + j + 1, tor - cnt,
                                               std::vector<int>(o.left_c.begin() + j + 1, o.left_c.end()), o.mem_c);
                        }
                        int rem = beg + j;
                        if (rem != static_cast<int>(vis.size()) - 1) {
                            o.mem_c.push_back(rem); // Maintain sequence order
                        }
                    }
                }
            }

            // Populate `other2` with results from `other`
            other2.resize(other.size());
            std::transform(other.begin(), other.end(), other2.begin(), [](const other_ &o) { return o.mem_c; });
        }

        // Filter elements in `segment` based on `mem`
        for (int i = 1; i < dim; ++i) {
            if (mem[i]) {
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        // Track which segments are visible in each sub-plan
        std::vector<ankerl::unordered_dense::set<int>> num_vis(dim);
        for (int i = 0; i < other2.size(); ++i) {
            for (int j : other2[i]) {
                for (int k : segment[j]) { num_vis[k].insert(i); }
            }
        }

        // Update `mem` based on `num_vis` and clear matched segments
        for (int i = 1; i < dim; ++i) {
            if (num_vis[i].size() == other2.size()) {
                mem.set(i);
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        // Prepare `mem_other` to hold unique memory states for sub-plans
        std::vector<std::pair<cutLong, std::vector<int>>> mem_other;
        for (const auto &i : other2) {
            cutLong p_mem;
            for (int j : i) {
                for (int k : segment[j]) { p_mem.set(k); }
            }

            // If no elements are set in `p_mem`, clear `plan` and exit
            if (p_mem.none()) {
                plan.clear();
                return;
            }

            // Add to `mem_other` only if no existing memory state matches
            bool found = false;
            for (auto &[existing_mem, mem_indices] : mem_other) {
                if (((p_mem & existing_mem) ^ p_mem).none()) {
                    existing_mem = p_mem;
                    mem_indices  = i;
                    found        = true;
                    break;
                }
            }
            if (!found) { mem_other.emplace_back(p_mem, i); }
        }

        // Transform `mem_other` into the output plan
        plan.resize(mem_other.size());
        std::transform(mem_other.begin(), mem_other.end(), plan.begin(),
                       [](const auto &entry) { return entry.second; });
    }
    void fillMemory() {
        if (!cuts.empty()) { throw std::runtime_error("cuts should be empty at first!"); }

        // Build v_r_map to count route occurrences for each vertex
        std::vector<ankerl::unordered_dense::map<int, int>> v_r_map(dim);
        for (int r_idx = 0; const auto &route : sol) {
            for (int j : route.route) { ++v_r_map[j][r_idx]; }
            ++r_idx;
        }

        int                 num_add_mem = 0;
        std::vector<double> vis_times(sol.size());

        for (const auto &r1c : old_cuts) {
            std::fill(vis_times.begin(), vis_times.end(), 0.0);

            const auto &plan  = map_rank1_multiplier[r1c.info_r1c.first.size()][r1c.info_r1c.second];
            const auto &coeff = std::get<0>(plan);
            int         deno  = std::get<1>(plan);
            double      rhs   = std::get<2>(plan);

            int cnt = 0;
            for (int v : r1c.info_r1c.first) {
                for (const auto &[route_idx, frequency] : v_r_map[v]) {
                    vis_times[route_idx] += frequency * coeff[cnt];
                }
                ++cnt;
            }

            std::transform(
                vis_times.begin(), vis_times.end(), sol.begin(), vis_times.begin(),
                [deno](double a, const auto &s) { return static_cast<int>(a / deno + tolerance) * s.frac_x; });

            double vio = std::accumulate(vis_times.begin(), vis_times.end(), -rhs);
            if (vio > tolerance) {
                cuts.push_back(r1c);
                cut_record[r1c.info_r1c.first].insert(r1c.info_r1c.second);
                ++num_add_mem;
            }
        }
    }

    void constructMemoryVertexBased() {
        std::vector<int> tmp_fill(dim);
        std::iota(tmp_fill.begin(), tmp_fill.end(), 0);

        for (auto &c : cuts) {
            ankerl::unordered_dense::set<int> mem;

            if (c.idx_r1c != INITIAL_IDX_R1C) {
                for (const auto &m : c.arc_mem) { mem.insert(m.second); }
            }

            bool  if_suc = false;
            auto &cut    = c.info_r1c;

            if (cut.second != 0) { findMemoryForRank1Multi(cut, mem, if_suc); }

            if (if_suc) {
                // Allocate memory positions for arc_mem
                c.arc_mem.clear();
                c.arc_mem.reserve(mem.size());

                for (int m : mem) { c.arc_mem.emplace_back(tmp_fill, m); }
            }
        }
    }

    void combinations(const std::vector<std::vector<std::vector<int>>>                  &array,
                      const std::vector<std::vector<ankerl::unordered_dense::set<int>>> &vec_segment, int i,
                      std::vector<int> &accum, const ankerl::unordered_dense::set<int> &mem, int &record_min,
                      ankerl::unordered_dense::set<int> &new_mem) {
        if (i == array.size()) {
            int  num     = 0;
            auto tmp_mem = mem;

            for (int j = 0; j < array.size(); ++j) {
                for (int k : array[j][accum[j]]) {
                    for (int l : vec_segment[j][k]) {
                        if (tmp_mem.insert(l).second) { // Insert and check if new element added
                            ++num;
                        }
                    }
                }
            }

            if (num < record_min) {
                record_min = num;
                new_mem    = std::move(tmp_mem);
            }
        } else {
            for (int j = 0; j < array[i].size(); ++j) {
                accum.push_back(j);
                combinations(array, vec_segment, i + 1, accum, mem, record_min, new_mem);
                accum.pop_back(); // Backtrack
            }
        }
    }

    void findMemoryForRank1Multi(const std::pair<std::vector<int>, int> &cut_pair,
                                 ankerl::unordered_dense::set<int> &mem, bool &if_suc) {
        if_suc               = true;
        const auto &cut      = cut_pair.first;
        int         plan_idx = cut_pair.second;
        int         size     = static_cast<int>(cut.size());

        const auto &multi       = std::get<0>(map_rank1_multiplier[size][plan_idx]);
        int         denominator = std::get<1>(map_rank1_multiplier[size][plan_idx]);

        // Calculate visit times based on `cut` and `multi`
        std::vector<int>                       num_vis_times(sol.size(), 0);
        ankerl::unordered_dense::map<int, int> map_cut_mul;
        for (int i = 0; i < cut.size(); ++i) {
            int multiplier      = multi[i];
            map_cut_mul[cut[i]] = multiplier;
            for (const auto &[route_idx, frequency] : v_r_map[cut[i]]) {
                num_vis_times[route_idx] += multiplier * frequency;
            }
        }
        std::transform(num_vis_times.begin(), num_vis_times.end(), num_vis_times.begin(),
                       [denominator](int x) { return x / denominator; });

        cutLong                                                     mem_long;
        std::vector<std::vector<std::vector<int>>>                  vec_data;
        std::vector<std::vector<ankerl::unordered_dense::set<int>>> vec_segment_route;

        // Populate `vec_data` and `vec_segment_route`
        for (int num = 0; const auto &route : sol) {
            if (num_vis_times[num++] == 0) continue;

            std::vector<int>                               vis;
            std::vector<ankerl::unordered_dense::set<int>> segment_route;
            ankerl::unordered_dense::set<int>              tmp_seg;

            for (int v : route.route) {
                if (auto it = map_cut_mul.find(v); it != map_cut_mul.end()) {
                    vis.push_back(it->second);
                    segment_route.push_back(std::move(tmp_seg));
                    tmp_seg.clear();
                } else {
                    tmp_seg.insert(v);
                }
            }
            if (!segment_route.empty()) segment_route.erase(segment_route.begin()); // Remove first segment

            std::vector<std::vector<int>> data;
            findPlanForRank1Multi(vis, denominator, mem_long, segment_route, data);
            if (!data.empty()) {
                vec_data.push_back(std::move(data));
                vec_segment_route.push_back(std::move(segment_route));
            }
        }

        // Filter `vec_data` and `vec_segment_route` based on `mem_long`
        for (size_t i = 0; i < vec_data.size();) {
            bool if_clear = false;
            for (const auto &segment : vec_data[i]) {
                bool all_satisfied = std::all_of(segment.begin(), segment.end(), [&](int idx) {
                    return std::all_of(vec_segment_route[i][idx].begin(), vec_segment_route[i][idx].end(),
                                       [&](int v) { return mem_long[v]; });
                });

                if (all_satisfied) {
                    vec_data.erase(vec_data.begin() + i);
                    vec_segment_route.erase(vec_segment_route.begin() + i);
                    if_clear = true;
                    break;
                }
            }
            if (!if_clear) ++i;
        }

        // Early exit if `mem_long` exceeds memory factor limit
        if (mem_long.count() > static_cast<int>(dim * max_cut_mem_factor)) {
            if_suc = false;
            return;
        }

        // Populate `mem` based on `mem_long`
        for (int i = 1; i < dim - 1; ++i) {
            if (mem_long[i]) mem.insert(i);
        }

        findMemAggressively(vec_data, vec_segment_route, mem);

        // Perform combinations optimization if `cnt > 1`
        if (std::accumulate(vec_data.begin(), vec_data.end(), 1,
                            [](size_t acc, const auto &data) { return acc * data.size(); }) > 1) {
            std::vector<int>                  tmp;
            ankerl::unordered_dense::set<int> new_mem;
            int                               record_min = std::numeric_limits<int>::max();
            combinations(vec_data, vec_segment_route, 0, tmp, mem, record_min, new_mem);
            mem = std::move(new_mem);
        }
    }

    void printPool() {
        for (const auto &i : rank1_multi_label_pool) {
            fmt::print("Cut: ");
            for (int node : i.c) { fmt::print("{} ", node); }
            fmt::print("| Plan Index: {}\n", i.plan_idx);
        }
    }

    void print_generated_rank1_multi_pool() {
        {
            for (const auto &[key, tuples] : generated_rank1_multi_pool) {
                std::cout << "Key: " << key << "\n";
                for (const auto &[bitset, int_val, double_val] : tuples) {
                    std::cout << "  Bitset: " << bitset << ", Int: " << int_val << ", Double: " << double_val << "\n";
                }
            }
        }
    }

    void findMemAggressively(const std::vector<std::vector<std::vector<int>>>                  &array,
                             const std::vector<std::vector<ankerl::unordered_dense::set<int>>> &vec_segment,
                             ankerl::unordered_dense::set<int>                                 &mem) {
        for (int i = 0; i < array.size(); ++i) {
            const auto      &r = array[i];
            std::vector<int> ele_size(r.size(), 0);

            // Calculate sizes of elements in `r` based on `vec_segment`
            for (int j = 0; j < r.size(); ++j) {
                for (int k : r[j]) { ele_size[j] += static_cast<int>(vec_segment[i][k].size()); }
            }

            // Find the index with the minimum size in `ele_size`
            int min_idx = std::distance(ele_size.begin(), std::min_element(ele_size.begin(), ele_size.end()));

            // Add elements from the minimum index segment to `mem`
            for (int k : r[min_idx]) { mem.insert(vec_segment[i][k].begin(), vec_segment[i][k].end()); }
        }
    }

private:
};
