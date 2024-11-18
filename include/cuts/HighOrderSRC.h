/*
 * @file HighDimCutsGenerator.h
 * @brief High-dimensional cuts generator for the VRP
 *
 * This class is responsible for generating high-dimensional cuts for the VRP.
 * The class is based on RouteOpt: https://github.com/Zhengzhong-You/RouteOpt/
 *
 */

#pragma once

#include "Common.h"
#include "Path.h"
#include "SparseMatrix.h"
#include "VRPNode.h"

#include <ranges>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "Bitset.h"
#include "utils/Hashes.h"

constexpr int INITIAL_RANK_1_MULTI_LABEL_POOL_SIZE              = 50;
using yzzLong                                                   = Bitset<N_SIZE>;
using cutLong                                                   = yzzLong;
constexpr double tolerance                                      = 1e-6;
constexpr int    max_row_rank1                                  = 5;
constexpr int    max_heuristic_initial_seed_set_size_row_rank1c = 5;
constexpr int    max_heuristic_sep_mem4_row_rank1               = 9;

constexpr int    max_num_r1c_per_round = 20;
constexpr double cut_vio_factor        = 0.1;

struct R1c {
    std::pair<std::vector<int>, int> info_r1c{}; // cut and plan
    int                              rhs{};
    std::vector<int>                 arc_mem{}; // Store only memory positions
};

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

// namespace py = pybind11;

class HighDimCutsGenerator {
public:
    double max_cut_mem_factor = 0.15;

    HighDimCutsGenerator(int dim, int maxRowRank, double tolerance)
        : dim(dim), max_row_rank1(maxRowRank), TOLERANCE(tolerance), num_label(0) {
        generateOptimalMultiplier();
        // printMultiplierMap();
    }

    void setMemFactor(double factor) { max_cut_mem_factor = factor; }

    void generateOptimalMultiplier();
    void printCuts();

    std::vector<R1c> &getCuts() { return cuts; }

    ankerl::unordered_dense::map<int, std::vector<std::tuple<std::vector<int>, int, int>>> map_rank1_multiplier;
    ankerl::unordered_dense::map<int, ankerl::unordered_dense::set<int>> map_rank1_multiplier_dominance{};
    std::vector<std::vector<std::vector<std::vector<int>>>>              record_map_rank1_combinations{};

    void printMultiplierMap();

    void generatePermutations(ankerl::unordered_dense::map<int, int> &count_map, std::vector<int> &result,
                              std::vector<std::vector<int>> &results, int remaining);

    SparseMatrix matrix;
    void         initialize(const std::vector<Path> &routes) {
        this->sol = routes;
        initialSupportvector();
        cacheValidRoutesAndMappings(); // Call new caching function
    }

    std::vector<VRPNode> nodes;
    void                 setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    std::vector<Path> sol;
    int               dim, max_row_rank1, num_label;
    double            TOLERANCE;
    // std::vector<ankerl::unordered_dense::map<int, int>>        v_r_map;
    gch::small_vector<std::vector<int>>                                                            v_r_map;
    gch::small_vector<std::pair<std::vector<int>, std::vector<int>>>                               c_N_noC;
    ankerl::unordered_dense::map<Bitset<N_SIZE>, std::vector<std::pair<std::vector<int>, double>>> map_cut_plan_vio;
    std::vector<Rank1MultiLabel>                                                            rank1_multi_label_pool;
    ankerl::unordered_dense::map<int, std::vector<std::tuple<Bitset<N_SIZE>, int, double>>> generated_rank1_multi_pool;
    std::vector<R1c>                                                                        cuts;
    std::vector<R1c>                                                                        old_cuts;

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
        cut_cache.clear();
        num_valid_routes = 0;
    }

    void getHighDimCuts() {
        constructVRMapAndSeedCrazy();
        startSeedCrazy();
        for (int i = 0; i < num_label;) { operationsCrazy(rank1_multi_label_pool[i], i); }
        constructCutsCrazy();

        // send to python
        // sendToPython(cuts, sol);
        old_cuts = cuts;
    }

    void processSeedForPlan(int plan_idx, std::vector<Rank1MultiLabel> &local_pool) {
        // Iterate through each `c, wc` pair in `c_N_noC`
        for (auto &[c, wc] : c_N_noC) {
            double vio, best_vio;
            exactFindBestPermutationForOnePlan(c, plan_idx, vio);
            if (vio < TOLERANCE) continue;

            best_vio                      = vio;
            char                best_oper = 'o';
            int                 add_j = -1, remove_j = -1;
            std::pair<int, int> swap_i_j = {-1, -1};

            // Perform add search and update best operation if necessary
            addSearchCrazy(plan_idx, c, wc, vio, add_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 'a';
            }

            // Perform remove search and update best operation if necessary
            removeSearchCrazy(plan_idx, c, vio, remove_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 'r';
            }

            // Perform swap search and update best operation if necessary
            swapSearchCrazy(plan_idx, c, wc, vio, swap_i_j);
            if (vio > best_vio) {
                best_vio  = vio;
                best_oper = 's';
            }

            // Prepare `new_c` and `new_w_no_c` based on the best operation
            std::vector<int> new_c, new_w_no_c;
            applyBestOperation(best_oper, c, wc, add_j, remove_j, swap_i_j, new_c, new_w_no_c, plan_idx, best_vio);

            // Only add to `local_pool` if an operation other than 'o' was performed
            if (best_oper != 'o') { local_pool.emplace_back(new_c, new_w_no_c, plan_idx, best_vio, best_oper); }
        }
    }

    void applyBestOperation(char best_oper, const std::vector<int> &c, const std::vector<int> &wc, int add_j,
                            int remove_j, const std::pair<int, int> &swap_i_j, std::vector<int> &new_c,
                            std::vector<int> &new_w_no_c, int plan_idx, double best_vio) {
        switch (best_oper) {
        case 'o': // No operation, directly add to target pool
            if (c.size() >= 2 && c.size() <= max_row_rank1) {
                cutLong tmp;
                for (int i : c) tmp.set(i);
                {
                    std::lock_guard<std::mutex> lock(pool_mutex);
                    generated_rank1_multi_pool[static_cast<int>(c.size())].emplace_back(tmp, plan_idx, best_vio);
                }
            }
            break;

        case 'a': // Add operation
            new_c = c;
            new_c.push_back(add_j);
            new_w_no_c.reserve(wc.size() - 1);
            std::copy_if(wc.begin(), wc.end(), std::back_inserter(new_w_no_c), [add_j](int w) { return w != add_j; });
            break;

        case 'r': // Remove operation
            new_c.reserve(c.size() - 1);
            std::copy_if(c.begin(), c.end(), std::back_inserter(new_c), [remove_j](int i) { return i != remove_j; });
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
                rank1_multi_label_pool[num_label++] = (label);
            }
        });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
    }

    gch::small_vector<double> cached_valid_routes;
    gch::small_vector<int>    cached_map_old_new_routes;
    int                       num_valid_routes;

    void cacheValidRoutesAndMappings() {
        cached_valid_routes.clear();
        cached_map_old_new_routes.clear();

        // Reserve space for the expected number of valid routes
        cached_valid_routes.reserve(sol.size());
        cached_map_old_new_routes.resize(sol.size(), -1); // Initialize all elements to -1

        for (int i = 0; i < sol.size(); ++i) {
            if (sol[i].frac_x > 1e-3) { // Only consider routes with frac_x > threshold
                cached_valid_routes.push_back(sol[i].frac_x);
                cached_map_old_new_routes[i] = static_cast<int>(cached_valid_routes.size() - 1);
                num_valid_routes++;
            }
            // No need for else clause as -1 already indicates invalid entries
        }
    }

    // declare map_cut_plan_vio_mutex
    std::mutex map_cut_plan_vio_mutex;
    std::mutex cut_cache_mutex;

    ankerl::unordered_dense::map<std::vector<int>, std::vector<std::vector<int>>, VectorIntHash> cut_cache;

    void exactFindBestPermutationForOnePlan(std::vector<int> &cut, const int plan_idx, double &vio) {
        const int   cut_size = static_cast<int>(cut.size());
        const auto &plan     = map_rank1_multiplier[cut_size][plan_idx];

        if (get<1>(plan) == 0) {
            vio = -std::numeric_limits<double>::max();
            return;
        }

        cutLong tmp = 0;
        for (const auto &it : cut) {
            if (it >= 0 && it < v_r_map.size()) {
                tmp.set(it); // Optimized bitset handling by reducing `v_r_map` size checks
            }
        }

        // Cache and minimize mutex usage
        bool found_in_cache = false;
        {
            std::lock_guard<std::mutex> lock(map_cut_plan_vio_mutex);
            if (auto it = map_cut_plan_vio.find(tmp);
                it != map_cut_plan_vio.end() && plan_idx < it->second.size() && !it->second[plan_idx].first.empty()) {
                vio            = it->second[plan_idx].second;
                found_in_cache = true;
            } else {
                map_cut_plan_vio[tmp].resize(7);
            }
        }
        if (found_in_cache) return;

        const int    denominator = get<1>(plan);
        const double rhs         = get<2>(plan);
        const auto  &coeffs      = record_map_rank1_combinations[cut_size][plan_idx];

        std::vector<int> map_r_numbers(sol.size(), 0); // Pre-allocate vector for route frequencies
        for (int idx : cut) {
            if (idx >= 0 && idx < v_r_map.size()) {
                for (int route_idx = 0; route_idx < v_r_map[idx].size(); ++route_idx) {
                    int frequency = v_r_map[idx][route_idx];
                    if (frequency > 0) {
                        map_r_numbers[route_idx] += frequency; // Direct indexing is faster than map access
                    }
                }
            }
        }
        for (int idx : cut) {
            if (idx >= 0 && idx < v_r_map.size()) {
                for (int route_idx = 0; route_idx < v_r_map[idx].size(); ++route_idx) {
                    int frequency = v_r_map[idx][route_idx];
                    if (frequency > 0) { // Only consider routes where the vertex has a frequency > 0
                        map_r_numbers[route_idx] += frequency;
                    }
                }
            }
        }

        static thread_local std::vector<std::vector<int>> cut_num_times_vis_routes;

        // Check if `cut` is already in cache
        {
            std::lock_guard<std::mutex> lock(cut_cache_mutex);
            if (auto cache_it = cut_cache.find(cut); cache_it != cut_cache.end()) {
                // Retrieve cached value if it exists
                cut_num_times_vis_routes = cache_it->second;
            } else {
                // Calculate and initialize `cut_num_times_vis_routes`
                cut_num_times_vis_routes.clear();
                cut_num_times_vis_routes.resize(cut_size);

                for (int i = 0; i < cut_size; ++i) {
                    int c = cut[i];
                    if (c >= 0 && c < v_r_map.size()) {
                        for (int route_idx = 0; route_idx < v_r_map[c].size(); ++route_idx) {
                            int frequency = v_r_map[c][route_idx];
                            if (frequency > 0) { // Only consider routes where the vertex has a frequency > 0
                                // print route_idx and cached_map_old_new_routes.size()
                                if (cached_map_old_new_routes[route_idx] != -1) {
                                    auto val = cached_map_old_new_routes[route_idx];
                                    cut_num_times_vis_routes[i].insert(cut_num_times_vis_routes[i].end(), frequency,
                                                                       val);
                                }
                            }
                        }
                    }
                }

                // Now, safely insert the newly computed `cut_num_times_vis_routes` into the cache
                cut_cache[cut] = cut_num_times_vis_routes;
            }
        }

        std::vector<double> num_times_vis_routes(num_valid_routes, 0.0);
        double              best_vio = -std::numeric_limits<double>::max();
        int                 best_idx = -1;

        // Loop through coeffs only once, using cumulative values for `num_times_vis_routes`
        for (size_t cnt = 0; cnt < coeffs.size(); ++cnt) {
            std::fill(num_times_vis_routes.begin(), num_times_vis_routes.end(), 0.0);

            for (int i = 0; i < cut_size; ++i) {
                // print cut_num_times_vis_routes size
                for (int j : cut_num_times_vis_routes[i]) {
                    // print j
                    num_times_vis_routes[j] += coeffs[cnt][i];
                }
            }

            double vio_tmp = -rhs;
            for (size_t i = 0; i < num_times_vis_routes.size(); ++i) {
                num_times_vis_routes[i] =
                    static_cast<int>(num_times_vis_routes[i] / denominator + tolerance) * cached_valid_routes[i];
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
            if (plan_idx < map_cut_plan_vio[tmp].size()) { map_cut_plan_vio[tmp][plan_idx] = {new_cut, vio}; }
        }
    }
    std::vector<std::vector<double>> cost_mat4_vertex;

    std::shared_ptr<HighDimCutsGenerator> clone() const {
        // Directly create a new instance without copying `pool_mutex`
        auto new_gen = std::shared_ptr<HighDimCutsGenerator>(new HighDimCutsGenerator(N_SIZE, 5, 1e-6));
        new_gen->setDistanceMatrix(cost_mat4_vertex);
        new_gen->old_cuts = this->old_cuts;
        return new_gen;
    }

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) { cost_mat4_vertex = distances; }

    void generateSepHeurMem4Vertex() {
        rank1_sep_heur_mem4_vertex.resize(dim);

        // Precompute half-costs for nodes to avoid repeated divisions
        std::vector<double> half_cost(dim);
        for (int i = 0; i < dim; ++i) { half_cost[i] = nodes[i].cost / 2; }

        for (int i = 0; i < dim; ++i) {
            // Initialize and populate the `cost` vector directly for each `i`
            std::vector<std::pair<int, double>> cost(dim);
            cost[0] = {0, INFINITY};

            for (int j = 1; j < dim - 1; ++j) { cost[j] = {j, cost_mat4_vertex[i][j] - (half_cost[i] + half_cost[j])}; }

            // Use partial sort to get only the top `max_heuristic_sep_mem4_row_rank1` elements
            std::partial_sort(cost.begin(), cost.begin() + max_heuristic_sep_mem4_row_rank1, cost.end(),
                              [](const auto &a, const auto &b) { return a.second < b.second; });

            // Set bits in `vst2` for the smallest costs
            cutLong &vst2 = rank1_sep_heur_mem4_vertex[i];
            for (int k = 0; k < max_heuristic_sep_mem4_row_rank1; ++k) { vst2.set(cost[k].first); }
        }
    }

    void constructVRMapAndSeedCrazy() {
        // Resize `rank1_sep_heur_mem4_vertex` and initialize `v_r_map`
        rank1_sep_heur_mem4_vertex.resize(dim);
        v_r_map.assign(dim, std::vector<int>(sol.size(), 0));

        // Populate `v_r_map` with route appearances for each vertex `i`
        for (int r = 0; r < sol.size(); ++r) {
            for (int i : sol[r].route) {
                if (i > 0 && i < dim - 1) { ++v_r_map[i][r]; }
            }
        }

        // Initialize `seed_map` with reserved space
        ankerl::unordered_dense::map<cutLong, cutLong> seed_map;
        seed_map.reserve(4096);

        // Cache dimensions for efficiency
        const int max_dim_minus_1 = dim - 1;

        // Iterate through vertices to build `seed_map`
        for (int i = 1; i < max_dim_minus_1; ++i) {
            if (v_r_map[i].empty()) continue;

            cutLong     wc;
            const auto &heur_mem = rank1_sep_heur_mem4_vertex[i];

            // Create bitset `wc` based on neighbors in routes containing vertex `i`
            for (int route_index = 0; route_index < v_r_map[i].size(); ++route_index) {
                if (v_r_map[i][route_index] > 0) {
                    for (int v : sol[route_index].route) {
                        if (v > 0 && v < max_dim_minus_1 && heur_mem.test(v)) { wc.set(v); }
                    }
                }
            }

            // Construct `tmp_c` for routes excluding vertex `i`
            for (int r = 0; r < sol.size(); ++r) {
                if (v_r_map[i][r] > 0) continue;

                cutLong tmp_c;
                for (int v : sol[r].route) {
                    if (v > 0 && v < max_dim_minus_1 && wc.test(v)) { tmp_c.set(v); }
                }
                tmp_c.set(i);

                // Add to `seed_map` if within size limits
                int c_size = static_cast<int>(tmp_c.count());
                if (c_size >= 4 && c_size <= max_heuristic_initial_seed_set_size_row_rank1c) {
                    auto [it, inserted] = seed_map.try_emplace(tmp_c, wc ^ tmp_c);
                    if (!inserted) { it->second |= wc ^ tmp_c; }
                }
            }
        }

        // Prepare `c_N_noC` by transforming `seed_map` entries
        c_N_noC.resize(seed_map.size());
        int cnt = 0;
        for (const auto &[fst, snd] : seed_map) {
            auto &[tmp_fst, tmp_snd] = c_N_noC[cnt];
            tmp_fst.reserve(fst.count());
            tmp_snd.reserve(snd.count());

            for (int i = 1; i < max_dim_minus_1; ++i) {
                if (fst.test(i)) {
                    tmp_fst.push_back(i);
                } else if (snd.test(i)) {
                    tmp_snd.push_back(i);
                }
            }
            ++cnt;
        }
    }

    ////////////////////////////////////////
    // Operators
    ////////////////////////////////////////
    inline void addSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c, double &new_vio,
                               int &add_j) {
        const int new_c_size = static_cast<int>(c.size()) + 1;

        // Validate and precompute the plan reference
        if (new_c_size > max_row_rank1 || new_c_size >= map_rank1_multiplier.size() ||
            plan_idx >= map_rank1_multiplier[new_c_size].size() ||
            !get<1>(map_rank1_multiplier[new_c_size][plan_idx])) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }
        const auto &plan = map_rank1_multiplier[new_c_size][plan_idx];

        std::vector<int> tmp_c = c; // Start with `c` and add one element
        tmp_c.push_back(0);         // Reserve last spot for the candidate in `w_no_c`

        double best_vio       = -std::numeric_limits<double>::max();
        int    best_candidate = -1;

        // Lambda for violation calculation
        auto calculateViolation = [&](int candidate) {
            tmp_c.back() = candidate;
            double vio;
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
            return vio;
        };

        // Loop through w_no_c to find the best candidate
        for (const int cplus : w_no_c) {
            double vio = calculateViolation(cplus);
            if (vio > best_vio) {
                best_vio       = vio;
                best_candidate = cplus;
            }
        }

        new_vio = best_vio - tolerance;
        add_j   = best_candidate;
    }

    inline void removeSearchCrazy(int plan_idx, const std::vector<int> &c, double &new_vio, int &remove_j) {
        const int new_c_size = static_cast<int>(c.size()) - 1;

        // Validate and precompute the plan reference
        if (new_c_size < 3 || new_c_size >= map_rank1_multiplier.size() ||
            plan_idx >= map_rank1_multiplier[new_c_size].size() ||
            !get<1>(map_rank1_multiplier[new_c_size][plan_idx])) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }
        const auto &plan = map_rank1_multiplier[new_c_size][plan_idx];

        std::vector<int> tmp_c(new_c_size); // Pre-resize tmp_c for removal operations
        double           best_vio       = -std::numeric_limits<double>::max();
        int              best_candidate = -1;

        // Lambda for violation calculation
        auto calculateViolation = [&](int remove_index) {
            std::copy(c.begin(), c.begin() + remove_index, tmp_c.begin());
            std::copy(c.begin() + remove_index + 1, c.end(), tmp_c.begin() + remove_index);
            double vio;
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
            return vio;
        };

        // Loop through indices in `c` to find the best candidate for removal
        for (int i = 0; i < c.size(); ++i) {
            double vio = calculateViolation(i);
            if (vio > best_vio) {
                best_vio       = vio;
                best_candidate = c[i];
            }
        }

        new_vio  = best_vio + tolerance; // Apply penalty
        remove_j = best_candidate;
    }

    inline void swapSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c,
                                double &new_vio, std::pair<int, int> &swap_i_j) {
        const int c_size = static_cast<int>(c.size());

        // Validate and precompute the plan reference
        if (c_size < 3 || c_size > max_row_rank1 || c_size >= map_rank1_multiplier.size() ||
            plan_idx >= map_rank1_multiplier[c_size].size() || !get<1>(map_rank1_multiplier[c_size][plan_idx])) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }
        const auto &plan = map_rank1_multiplier[c_size][plan_idx];

        std::vector<int>    tmp_c     = c; // Start with a copy of `c` to swap elements
        double              best_vio  = -std::numeric_limits<double>::max();
        std::pair<int, int> best_swap = {-1, -1};

        // Lambda for violation calculation
        auto calculateViolation = [&](int index, int candidate) {
            tmp_c[index] = candidate;
            double vio;
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
            return vio;
        };

        // Loop through each element in `c` and each candidate in `w_no_c`
        for (int i = 0; i < c.size(); ++i) {
            int original = tmp_c[i];
            for (int swap_candidate : w_no_c) {
                if (swap_candidate == original) continue;

                double vio = calculateViolation(i, swap_candidate);
                if (vio > best_vio) {
                    best_vio  = vio;
                    best_swap = {original, swap_candidate};
                }
            }
            tmp_c[i] = original; // Restore original element
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

        std::array<std::pair<int, double>, 4> move_vio = {{{0, vio},
                                                           {1, -std::numeric_limits<double>::max()},
                                                           {2, -std::numeric_limits<double>::max()},
                                                           {3, -std::numeric_limits<double>::max()}}};

        bool                               add_checked = false, remove_checked = false, swap_checked = false;
        std::optional<int>                 add_j, remove_j;
        std::optional<std::pair<int, int>> swap_i_j;
        double                             new_vio = -std::numeric_limits<double>::max();

        // Determine which moves to evaluate based on `dir`
        if (dir == 'a' || dir == 's') {
            add_j.emplace(); // Emplace a default int for add_j
            addSearchCrazy(plan_idx, new_cij, w_no_cij, new_vio, *add_j);
            move_vio[1].second = new_vio;
            add_checked        = true;
        }
        if (dir == 'r' || dir == 's') {
            remove_j.emplace(); // Emplace a default int for remove_j
            removeSearchCrazy(plan_idx, new_cij, new_vio, *remove_j);
            move_vio[2].second = new_vio;
            remove_checked     = true;
        }
        if (dir == 'a' || dir == 'r') {
            swap_i_j.emplace(); // Emplace a default pair<int, int> for swap_i_j
            swapSearchCrazy(plan_idx, new_cij, w_no_cij, new_vio, *swap_i_j);
            move_vio[3].second = new_vio;
            swap_checked       = true;
        }

        // Find the best move by maximizing violation score
        auto   best_move_it  = std::max_element(move_vio.begin(), move_vio.end(),
                                                [](const auto &a, const auto &b) { return a.second < b.second; });
        int    best_move     = best_move_it->first;
        double best_move_vio = best_move_it->second;

        // Handle no operation (early exit)
        if (best_move == 0) {
            cutLong tmp;
            for (int j : new_cij) tmp.set(j);
            generated_rank1_multi_pool[static_cast<int>(new_cij.size())].emplace_back(tmp, plan_idx, best_move_vio);
            ++i;
            return;
        }

        // Execute the chosen operation
        switch (best_move) {
        case 1: // Add operation
            if (add_checked && add_j.has_value()) {
                w_no_cij.erase(std::remove(w_no_cij.begin(), w_no_cij.end(), *add_j), w_no_cij.end());
                new_cij.push_back(*add_j);
            }
            break;

        case 2: // Remove operation
            if (remove_checked && remove_j.has_value()) {
                new_cij.erase(std::remove(new_cij.begin(), new_cij.end(), *remove_j), new_cij.end());
            }
            break;

        case 3: // Swap operation
            if (swap_checked && swap_i_j.has_value()) {
                auto pos = std::find(new_cij.begin(), new_cij.end(), swap_i_j->first);
                if (pos != new_cij.end()) {
                    *pos = swap_i_j->second;
                    w_no_cij.erase(std::remove(w_no_cij.begin(), w_no_cij.end(), swap_i_j->second), w_no_cij.end());
                }
            }
            break;

        default: throw std::runtime_error("Invalid best move");
        }

        vio = best_move_vio;
    }

    void chooseCuts(const std::vector<R1c> &tmp_cuts, std::vector<R1c> &chosen_cuts, int numCuts) {
        numCuts = std::min(numCuts, static_cast<int>(tmp_cuts.size()));
        if (numCuts == 0) return;

        chosen_cuts.reserve(numCuts); // Reserve space for the chosen cuts to avoid repeated reallocations

        for (const auto &cut : tmp_cuts) {
            const auto &fst = cut.info_r1c.first;
            const auto &snd = cut.info_r1c.second;

            // Attempt to insert in cut_record and skip if already exists
            // if (!cut_record[fst].emplace(snd).second) continue;

            int         size  = static_cast<int>(fst.size());
            const auto &coeff = get<0>(map_rank1_multiplier[size][snd]);

            // Resize and clear `tmp_cut` for current cut's maximum coefficient value
            std::vector<std::vector<int>> tmp_cut(coeff[0] + 1);

            // Populate `tmp_cut` based on `fst` and `coeff`
            for (int i = 0; i < size; ++i) { tmp_cut[coeff[i]].push_back(fst[i]); }

            // Sort each group in `tmp_cut` if needed
            for (auto &group : tmp_cut) {
                if (group.size() > 1) { pdqsort(group.begin(), group.end()); }
            }

            // Flatten `tmp_cut` into `new_cut`, inserting groups in descending order of coefficients
            std::vector<int> new_cut;
            new_cut.reserve(size);
            for (int i = static_cast<int>(tmp_cut.size()) - 1; i >= 0; --i) {
                new_cut.insert(new_cut.end(), tmp_cut[i].begin(), tmp_cut[i].end());
            }

            // Add the new cut to chosen cuts
            chosen_cuts.emplace_back();
            chosen_cuts.back().info_r1c = std::make_pair(std::move(new_cut), snd);

            // Stop if the required number of cuts is reached
            if (--numCuts == 0) break;
        }
    }

    void constructCutsCrazy() {
        ankerl::unordered_dense::set<cutLong> cut_set; // Reuse across pools if keys/plans are shared
        ankerl::unordered_dense::set<int>     p_set;

        for (auto &pool : generated_rank1_multi_pool) {
            if (pool.second.empty()) continue; // Skip empty pools

            // Sort cuts in descending order of violation score
            pdqsort(pool.second.begin(), pool.second.end(),
                    [](const auto &a, const auto &b) { return std::get<2>(a) > std::get<2>(b); });

            const double vio_threshold = std::get<2>(pool.second.front()) * cut_vio_factor;

            // Reserve memory for cuts up to the maximum allowed per round
            std::vector<R1c> tmp_cuts;
            tmp_cuts.reserve(std::min(pool.second.size(), static_cast<size_t>(max_num_r1c_per_round)));

            int num_cuts = 0;
            for (const auto &cut : pool.second) {
                if (std::get<2>(cut) < vio_threshold) break; // Stop if violation score falls below threshold

                const auto &key      = std::get<0>(cut);
                int         plan_idx = std::get<1>(cut);

                // Only proceed if the cut is unique in `cut_set` and `p_set`
                if (!cut_set.contains(key) || !p_set.contains(plan_idx)) {
                    const auto &cut_plan = map_cut_plan_vio[key][plan_idx];
                    tmp_cuts.emplace_back(R1c{std::make_pair(cut_plan.first, plan_idx)});

                    cut_set.insert(key);
                    p_set.insert(plan_idx);
                    ++num_cuts;

                    if (num_cuts >= max_num_r1c_per_round) break; // Stop if the limit is reached
                }
            }

            // Move the selected cuts to the main cuts vector in a single operation
            chooseCuts(tmp_cuts, cuts, max_num_r1c_per_round);
        }
    }

    void findPlanForRank1Multi(const std::vector<int> &vis, const int denominator, cutLong &mem,
                               std::vector<ankerl::unordered_dense::set<int>> &segment,
                               std::vector<std::vector<int>>                  &plan) {
        int sum = std::accumulate(vis.begin(), vis.end(), 0);
        int mod = sum % denominator;

        std::vector<int> key(vis);
        key.push_back(mod); // Add mod as the last element

        auto &other2 = rank1_multi_mem_plan_map[key];
        if (other2.empty()) {
            std::deque<std::tuple<int, int, std::vector<int>, std::vector<int>>> states;
            states.emplace_back(0, mod, vis, std::vector<int>{});

            while (!states.empty()) {
                auto [beg, tor, left_c, mem_c] = std::move(states.front());
                states.pop_front();

                int cnt = 0;
                for (size_t j = 0; j < left_c.size(); ++j) {
                    cnt += left_c[j];
                    cnt %= denominator;

                    if (cnt > 0) {
                        if (cnt <= tor && j + 1 < left_c.size()) {
                            states.emplace_back(beg + j + 1, tor - cnt,
                                                std::vector<int>(left_c.begin() + j + 1, left_c.end()), mem_c);
                        }
                        int rem = beg + j;
                        if (rem != vis.size() - 1) { mem_c.push_back(rem); }
                    }
                }
                other2.push_back(std::move(mem_c));
            }
        }

        for (int i = 1; i < dim; ++i) {
            if (mem[i]) {
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        std::vector<ankerl::unordered_dense::set<int>> num_vis(dim);
        for (size_t i = 0; i < other2.size(); ++i) {
            for (int j : other2[i]) {
                for (int k : segment[j]) { num_vis[k].insert(i); }
            }
        }

        for (int i = 1; i < dim; ++i) {
            if (num_vis[i].size() == other2.size()) {
                mem.set(i);
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        std::vector<std::pair<cutLong, std::vector<int>>> mem_other;
        for (const auto &i : other2) {
            cutLong p_mem;
            for (int j : i) {
                for (int k : segment[j]) { p_mem.set(k); }
            }

            if (p_mem.none()) {
                plan.clear();
                return;
            }

            bool found = false;
            for (auto &[existing_mem, mem_indices] : mem_other) {
                if (((p_mem & existing_mem) ^ p_mem).none()) {
                    existing_mem = std::move(p_mem);
                    mem_indices  = i;
                    found        = true;
                    break;
                }
            }
            if (!found) { mem_other.emplace_back(std::move(p_mem), i); }
        }

        plan.resize(mem_other.size());
        std::transform(mem_other.begin(), mem_other.end(), plan.begin(),
                       [](const auto &entry) { return entry.second; });
    }

    /*
        void fillMemory() {
            // Build v_r_map to count route occurrences for each vertex
            std::vector<ankerl::unordered_dense::map<int, int>> v_r_map(dim);
            for (int r_idx = 0; const auto &route : sol) {
                for (int j : route.route) { ++v_r_map[j][r_idx]; }
                ++r_idx;
            }

            int                 num_add_mem = 0;
            std::vector<double> vis_times(sol.size());

            for (const auto &r1c : old_cuts) {
                std::fill(vis_times.begin(), vis_times.end(), 0.0); // Reset vis_times for this cut

                // Fetch the rank-1 multiplier plan components
                const auto &[coeff, deno, rhs] =
       map_rank1_multiplier.at(r1c.info_r1c.first.size()).at(r1c.info_r1c.second);

                // Accumulate visibility times for each route
                int cnt = 0;
                for (int v : r1c.info_r1c.first) {
                    const auto &vertex_routes = v_r_map[v];
                    double      coeff_value   = coeff[cnt++];
                    for (const auto &[route_idx, frequency] : vertex_routes) {
                        vis_times[route_idx] += frequency * coeff_value;
                    }
                }

                // Transform vis_times based on denominator and tolerance
                std::transform(
                    vis_times.begin(), vis_times.end(), sol.begin(), vis_times.begin(),
                    [deno](double a, const auto &s) { return static_cast<int>(a / deno + tolerance) * s.frac_x; });

                // Calculate violation
                double vio = std::accumulate(vis_times.begin(), vis_times.end(), -rhs);
                if (vio > tolerance) {
                    cuts.push_back(r1c);
                    cut_record[r1c.info_r1c.first].insert(r1c.info_r1c.second);
                    ++num_add_mem;
                }
            }
        }
    */

    void constructMemoryVertexBased() {
        std::vector<int> tmp_fill(dim);
        std::iota(tmp_fill.begin(), tmp_fill.end(), 0);

        ankerl::unordered_dense::set<int> mem;
        mem.reserve(dim); // Reserve based on expected size to reduce reallocations

        for (auto &c : cuts) {
            mem.clear(); // Clear `mem` at the start of each iteration to reuse it

            bool  if_suc = false;
            auto &cut    = c.info_r1c;

            // Call `findMemoryForRank1Multi` only if `cut.second` is non-zero
            if (cut.second != 0) { findMemoryForRank1Multi(cut, mem, if_suc); }

            // Only assign to `arc_mem` if `if_suc` is true
            if (if_suc) {
                c.arc_mem.assign(mem.begin(), mem.end()); // Assign elements of `mem` to `arc_mem`
            }
        }
    }

    void combinations(const std::vector<std::vector<std::vector<int>>>                  &array,
                      const std::vector<std::vector<ankerl::unordered_dense::set<int>>> &vec_segment, int i,
                      std::vector<int> &accum, const ankerl::unordered_dense::set<int> &mem, int &record_min,
                      ankerl::unordered_dense::set<int> &new_mem) {
        if (i == array.size()) {
            int                               num     = 0;
            ankerl::unordered_dense::set<int> tmp_mem = mem; // Copy `mem` to track the new elements in this combination
            tmp_mem.reserve(mem.size() + 10);                // Reserve memory to reduce reallocations

            for (int j = 0; j < array.size(); ++j) {
                for (int k : array[j][accum[j]]) {
                    for (int l : vec_segment[j][k]) {
                        if (tmp_mem.insert(l).second) { // Only count if a new element was inserted
                            ++num;
                            if (num >= record_min) {
                                return; // Early exit if this path already exceeds the minimum
                            }
                        }
                    }
                }
            }

            // Update `record_min` and `new_mem` if a new minimum was found
            if (num < record_min) {
                record_min = num;
                new_mem    = std::move(tmp_mem); // Use move semantics for efficiency
            }
        } else {
            // Iterate through choices for the current level `i`
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
            int vertex          = cut[i];
            int multiplier      = multi[i];
            map_cut_mul[vertex] = multiplier;

            for (int route_idx = 0; route_idx < v_r_map[vertex].size(); ++route_idx) {
                int frequency = v_r_map[vertex][route_idx];
                if (frequency > 0) { // Only proceed if the vertex appears in this route
                    num_vis_times[route_idx] += multiplier * frequency;
                }
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

    void findMemAggressively(const std::vector<std::vector<std::vector<int>>>                  &array,
                             const std::vector<std::vector<ankerl::unordered_dense::set<int>>> &vec_segment,
                             ankerl::unordered_dense::set<int>                                 &mem) {
        mem.reserve(array.size() * 10); // Adjust reserve size if more precise estimates are possible

        for (int i = 0; i < array.size(); ++i) {
            const auto &r        = array[i];
            const auto &segments = vec_segment[i];

            // Cache sizes to avoid recalculating in the inner loop
            std::vector<int> segment_sizes(r.size());
            for (int j = 0; j < r.size(); ++j) {
                segment_sizes[j] = std::accumulate(r[j].begin(), r[j].end(), 0, [&](int sum, int k) {
                    return sum + static_cast<int>(segments[k].size());
                });
            }

            // Find the index of the minimum size segment
            int min_idx = std::distance(segment_sizes.begin(), std::ranges::min_element(segment_sizes));

            // Manually insert elements from the minimum index segment into `mem`
            for (int k : r[min_idx]) { mem.insert(segments[k].begin(), segments[k].end()); }
        }
    }

private:
};
