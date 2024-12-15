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
constexpr int INITIAL_POOL_SIZE                                 = 100;
using yzzLong                                                   = Bitset<N_SIZE>;
using cutLong                                                   = yzzLong;
constexpr double tolerance                                      = 1e-6;
constexpr int    max_row_rank1                                  = 5;
constexpr int    max_heuristic_initial_seed_set_size_row_rank1c = 6;

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

class HighDimCutsGenerator {
public:
    int max_heuristic_sep_mem4_row_rank1 = 8;

    double max_cut_mem_factor = 0.15;

    template <typename T, std::size_t Alignment = 32>
    struct aligned_allocator {
        typedef T value_type;

        aligned_allocator() noexcept {}

        template <class U>
        aligned_allocator(const aligned_allocator<U, Alignment> &) noexcept {}

        template <typename U>
        struct rebind {
            typedef aligned_allocator<U, Alignment> other;
        };

        T *allocate(std::size_t n) {
            if (n == 0) return nullptr;
            void *ptr = std::aligned_alloc(Alignment, n * sizeof(T));
            if (!ptr) throw std::bad_alloc();
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, std::size_t) noexcept { std::free(p); }
    };

    HighDimCutsGenerator(int dim, int maxRowRank, double tolerance)
        : dim(dim), max_row_rank1(maxRowRank), TOLERANCE(tolerance), num_label(0) {
        cuts.reserve(INITIAL_POOL_SIZE);
        rank1_multi_label_pool.reserve(INITIAL_POOL_SIZE);
        generated_rank1_multi_pool.reserve(INITIAL_POOL_SIZE);
        map_cut_plan_vio.reserve(INITIAL_POOL_SIZE);

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

    void clearMemory() {
        cut_record.clear();
        v_r_map.clear();
        c_N_noC.clear();
        map_cut_plan_vio.clear();
        generated_rank1_multi_pool.clear();
        rank1_multi_label_pool.clear();
        rank1_multi_mem_plan_map.clear();
        cuts.clear();
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

    ankerl::unordered_dense::map<std::vector<int>, std::vector<std::vector<int>>, VectorIntHash> cut_cache;

    // declare map_cut_plan_vio_mutex
    std::shared_mutex map_cut_plan_vio_mutex;
    std::shared_mutex cut_cache_mutex;

    // For the frequency accumulation loop:
    void accumulate_frequencies_simd(int *map_r_numbers, const auto &route_freqs, int route_size) {
        const int simd_size = 8; // AVX2 processes 8 integers at once
        int       i         = 0;

        // Process 8 elements at a time using AVX2
        for (; i + simd_size <= route_size; i += simd_size) {
            __m256i freq = _mm256_loadu_si256((__m256i *)&route_freqs[i]);
            __m256i curr = _mm256_loadu_si256((__m256i *)&map_r_numbers[i]);
            __m256i mask = _mm256_cmpgt_epi32(freq, _mm256_setzero_si256());
            freq         = _mm256_and_si256(freq, mask);
            curr         = _mm256_add_epi32(curr, freq);
            _mm256_storeu_si256((__m256i *)&map_r_numbers[i], curr);
        }

        // Handle remaining elements
        for (; i < route_size; i++) {
            const int frequency = route_freqs[i];
            map_r_numbers[i] += frequency * (frequency > 0);
        }
    }

    void exactFindBestPermutationForOnePlan(std::vector<int> &cut, const int plan_idx, double &vio) {
        const int   cut_size = static_cast<int>(cut.size());
        const auto &plan     = map_rank1_multiplier[cut_size][plan_idx];

        if (get<1>(plan) == 0) {
            vio = -std::numeric_limits<double>::max();
            return;
        }

        // Use thread_local for frequently reused vectors to avoid allocations
        static thread_local cutLong tmp;
        tmp.reset();

        // SIMD-friendly loop with no branching for setting bits
        for (int i = 0; i < cut_size; ++i) {
            const int it = cut[i];
            tmp.set(it * (it >= 0 && it < v_r_map.size())); // Branchless using multiplication
        }

        // Quick cache lookup with shared lock
        {
            std::shared_lock<std::shared_mutex> lock(map_cut_plan_vio_mutex);
            if (auto it = map_cut_plan_vio.find(tmp);
                it != map_cut_plan_vio.end() && plan_idx < it->second.size() && !it->second[plan_idx].first.empty()) {
                vio = it->second[plan_idx].second;
                return;
            }
        }

        // Upgrade to exclusive lock to modify
        {
            std::unique_lock<std::shared_mutex> lock(map_cut_plan_vio_mutex);
            map_cut_plan_vio[tmp].resize(7);
        }

        const int    denominator = get<1>(plan);
        const double rhs         = get<2>(plan);
        const auto  &coeffs      = record_map_rank1_combinations[cut_size][plan_idx];

        // Use thread_local vectors to avoid allocations
        // static thread_local std::vector<int> map_r_numbers;
        alignas(32) static thread_local std::vector<int, aligned_allocator<int>> map_r_numbers;

        map_r_numbers.resize(sol.size());
        std::fill(map_r_numbers.begin(), map_r_numbers.end(), 0);

#ifndef __AVX2__
        // Pre-compute route frequencies in a single pass
        for (int idx : cut) {
            if (idx >= 0 && idx < v_r_map.size()) {
                const auto &route_freqs = v_r_map[idx];
                const int   route_size  = static_cast<int>(route_freqs.size());

                // SIMD-friendly loop for frequency accumulation
                for (int route_idx = 0; route_idx < route_size; ++route_idx) {
                    const int frequency = route_freqs[route_idx];
                    map_r_numbers[route_idx] += frequency * (frequency > 0); // Branchless using multiplication
                }
            }
        }
#else
        for (int idx : cut) {
            if (idx >= 0 && idx < v_r_map.size()) {
                const auto &route_freqs = v_r_map[idx];
                const int   route_size  = static_cast<int>(route_freqs.size());
                accumulate_frequencies_simd(map_r_numbers.data(), route_freqs, route_size);
            }
        }
#endif

        static thread_local std::vector<std::vector<int>> cut_num_times_vis_routes;

        // Cache lookup with minimized critical section
        bool cache_hit = false;
        {
            std::shared_lock<std::shared_mutex> lock(cut_cache_mutex);
            if (auto cache_it = cut_cache.find(cut); cache_it != cut_cache.end()) {
                cut_num_times_vis_routes = cache_it->second;
                cache_hit                = true;
            }
        }

        if (!cache_hit) {
            cut_num_times_vis_routes.clear();
            cut_num_times_vis_routes.resize(cut_size);

            // Pre-reserve space to avoid reallocations
            for (auto &route_vec : cut_num_times_vis_routes) {
                route_vec.reserve(num_valid_routes * 2); // Estimate capacity
            }

            for (int i = 0; i < cut_size; ++i) {
                const int c = cut[i];
                if (c >= 0 && c < v_r_map.size()) {
                    auto       &current_route = cut_num_times_vis_routes[i];
                    const auto &route_freqs   = v_r_map[c];

                    // Vectorizable loop for route processing
                    for (int route_idx = 0; route_idx < route_freqs.size(); ++route_idx) {
                        const int frequency = route_freqs[route_idx];
                        if (frequency > 0 && cached_map_old_new_routes[route_idx] != -1) {
                            const int val = cached_map_old_new_routes[route_idx];
                            current_route.insert(current_route.end(), frequency, val);
                        }
                    }
                }
            }

            // Update cache
            std::unique_lock<std::shared_mutex> lock(cut_cache_mutex);
            cut_cache[cut] = cut_num_times_vis_routes;
        }

        // Use thread_local vector for accumulation
        static thread_local std::vector<double> num_times_vis_routes;
        num_times_vis_routes.resize(num_valid_routes);

        double best_vio = -std::numeric_limits<double>::max();
        int    best_idx = -1;

        // SIMD-friendly loop for coefficient processing
        for (size_t cnt = 0; cnt < coeffs.size(); ++cnt) {
            std::fill(num_times_vis_routes.begin(), num_times_vis_routes.end(), 0.0);

            // Vectorizable nested loops
            for (int i = 0; i < cut_size; ++i) {
                const auto  &routes = cut_num_times_vis_routes[i];
                const double coeff  = coeffs[cnt][i];

                for (const int j : routes) { num_times_vis_routes[j] += coeff; }
            }

            // Compute violation score with SIMD-friendly loop
            double vio_tmp = -rhs;
            for (size_t i = 0; i < num_times_vis_routes.size(); ++i) {
                const double scaled_value =
                    static_cast<int>(num_times_vis_routes[i] / denominator + tolerance) * cached_valid_routes[i];
                vio_tmp += scaled_value;
            }

            if (vio_tmp > best_vio) {
                best_vio = vio_tmp;
                best_idx = static_cast<int>(cnt);
            }
        }
        vio = best_vio;

        // Final result preparation with thread_local storage
        static thread_local std::vector<std::pair<int, int>> cut_coeff;
        cut_coeff.resize(cut_size);

        for (int i = 0; i < cut_size; ++i) {
            if (best_idx >= 0 && best_idx < coeffs.size() && i < coeffs[best_idx].size()) {
                cut_coeff[i] = {cut[i], coeffs[best_idx][i]};
            }
        }

        pdqsort(cut_coeff.begin(), cut_coeff.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

        static thread_local std::vector<int> new_cut;
        new_cut.resize(cut_size);
        std::transform(cut_coeff.begin(), cut_coeff.end(), new_cut.begin(), [](const auto &a) { return a.first; });

        // Update final results
        std::unique_lock<std::shared_mutex> lock(map_cut_plan_vio_mutex);
        if (plan_idx < map_cut_plan_vio[tmp].size()) { map_cut_plan_vio[tmp][plan_idx] = {new_cut, vio}; }
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
        const int  max_dim_minus_1 = dim - 1;
        const auto sol_size        = sol.size();

        // Populate `v_r_map` with route appearances for each vertex `i`
#ifdef __cpp_lib_parallel_algorithm
        std::for_each(std::execution::par_unseq, sol.begin(), sol.end(), [&](const auto &route_data) {
            const auto r = &route_data - &sol[0]; // Get index without separate counter
            for (const int i : route_data.route) {
                if (i > 0 && i < max_dim_minus_1) { ++v_r_map[i][r]; }
            }
        });
#else
        for (size_t r = 0; r < sol_size; ++r) {
            for (const int i : sol[r].route) {
                if (i > 0 && i < max_dim_minus_1) { ++v_r_map[i][r]; }
            }
        }
#endif

        // Initialize `seed_map` with reserved space
        ankerl::unordered_dense::map<cutLong, cutLong> seed_map;
        seed_map.reserve(4096);

        // Cache dimensions for efficiency

        // Process vertices and build seed map
        for (size_t i = 1; i < max_dim_minus_1; ++i) {
            if (v_r_map[i].empty()) continue;

            // Create working set for current vertex
            cutLong     working_set;
            const auto &heur_mem = rank1_sep_heur_mem4_vertex[i];

            // Build working set from routes containing vertex i
            for (size_t route_idx = 0; route_idx < v_r_map[i].size(); ++route_idx) {
                if (v_r_map[i][route_idx] <= 0) continue;

                for (const int v : sol[route_idx].route) {
                    if (v > 0 && v < max_dim_minus_1 && heur_mem.test(v)) { working_set.set(v); }
                }
            }

            // Process routes not containing vertex i
            for (size_t r = 0; r < sol_size; ++r) {
                if (v_r_map[i][r] > 0) continue;

                cutLong candidate_set;
                for (const int v : sol[r].route) {
                    if (v > 0 && v < max_dim_minus_1 && working_set.test(v)) { candidate_set.set(v); }
                }
                candidate_set.set(i);

                const auto c_size = candidate_set.count();
                if (c_size >= 4 && c_size <= max_heuristic_initial_seed_set_size_row_rank1c) {
                    const auto complement = working_set ^ candidate_set;
                    auto [it, inserted]   = seed_map.try_emplace(candidate_set, complement);
                    if (!inserted) { it->second |= complement; }
                }
            }
        }

        // Prepare `c_N_noC` by transforming `seed_map` entries
        c_N_noC.resize(seed_map.size());
        std::transform(seed_map.begin(), seed_map.end(), c_N_noC.begin(), [max_dim_minus_1](const auto &seed_pair) {
            const auto &[fst, snd] = seed_pair;
            std::pair<std::vector<int>, std::vector<int>> result;

            result.first.reserve(fst.count());
            result.second.reserve(snd.count());

            for (size_t i = 1; i < max_dim_minus_1; ++i) {
                if (fst.test(i)) {
                    result.first.push_back(static_cast<int>(i));
                } else if (snd.test(i)) {
                    result.second.push_back(static_cast<int>(i));
                }
            }

            return result;
        });
    }

    ////////////////////////////////////////
    // Operators
    ////////////////////////////////////////
    // Common validation helper
    struct PlanValidationResult {
        bool                                          is_valid;
        const std::tuple<std::vector<int>, int, int> *plan;

        static PlanValidationResult
        validate(int plan_idx, int size,
                 const ankerl::unordered_dense::map<int, std::vector<std::tuple<std::vector<int>, int, int>>> &map) {
            auto size_it = map.find(size);
            if (size < 3 || size_it == map.end() || plan_idx >= size_it->second.size()) { return {false, nullptr}; }
            return {true, &size_it->second[plan_idx]};
        }
    };

    // Helper for finding best violation
    template <typename F>
    auto findBestViolation(F &&calculate_violation, const std::vector<int> &candidates,
                           double initial_vio = -std::numeric_limits<double>::max()) {
        double best_vio       = initial_vio;
        int    best_candidate = -1;

        for (const int candidate : candidates) {
            if (double vio = calculate_violation(candidate); vio > best_vio) {
                best_vio       = vio;
                best_candidate = candidate;
            }
        }

        return std::make_pair(best_vio, best_candidate);
    }

    inline void addSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c, double &new_vio,
                               int &add_j) {
        const int new_size = c.size() + 1;

        // Validate size and plan
        if (new_size > max_row_rank1) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        auto validation = PlanValidationResult::validate(plan_idx, new_size, map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        // Prepare temporary vector once
        std::vector<int> tmp_c = c;
        tmp_c.push_back(0);

        // Find best candidate
        auto [best_vio, best_candidate] = findBestViolation(
            [&](int candidate) {
                tmp_c.back() = candidate;
                double vio;
                exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
                return vio;
            },
            w_no_c);

        new_vio = best_vio - tolerance;
        add_j   = best_candidate;
    }

    inline void removeSearchCrazy(int plan_idx, const std::vector<int> &c, double &new_vio, int &remove_j) {
        const int new_size = c.size() - 1;

        // Validate size and plan
        auto validation = PlanValidationResult::validate(plan_idx, new_size, map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        // Prepare temporary vector once
        std::vector<int> tmp_c(new_size);
        std::vector<int> indices(c.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Find best removal
        auto [best_vio, best_idx] = findBestViolation(
            [&](int idx) {
                std::copy(c.begin(), c.begin() + idx, tmp_c.begin());
                std::copy(c.begin() + idx + 1, c.end(), tmp_c.begin() + idx);
                double vio;
                exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
                return vio;
            },
            indices);

        new_vio  = best_vio + tolerance;
        remove_j = best_idx >= 0 ? c[best_idx] : -1;
    }

    inline void swapSearchCrazy(int plan_idx, const std::vector<int> &c, const std::vector<int> &w_no_c,
                                double &new_vio, std::pair<int, int> &swap_i_j) {
        // Validate size and plan
        auto validation = PlanValidationResult::validate(plan_idx, c.size(), map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        std::vector<int>    tmp_c    = c;
        double              best_vio = -std::numeric_limits<double>::max();
        std::pair<int, int> best_swap{-1, -1};

        // Optimize loop structure to reduce vector modifications
        for (int i = 0; i < c.size(); ++i) {
            const int original = c[i];

            // Find best swap for current position
            auto [pos_best_vio, best_candidate] = findBestViolation(
                [&](int candidate) {
                    if (candidate == original) return -std::numeric_limits<double>::max();
                    tmp_c[i] = candidate;
                    double vio;
                    exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);
                    tmp_c[i] = original; // Restore immediately
                    return vio;
                },
                w_no_c);

            if (pos_best_vio > best_vio) {
                best_vio  = pos_best_vio;
                best_swap = {original, best_candidate};
            }
        }

        new_vio  = best_vio;
        swap_i_j = best_swap;
    }

    ////////////////////////////////////////

    struct MoveResult {
        double                                                 violation_score;
        std::variant<std::monostate, int, std::pair<int, int>> operation_data;

        MoveResult(double score = -std::numeric_limits<double>::max()) : violation_score(score) {}
    };

    void operationsCrazy(Rank1MultiLabel &label, int &i) {
        constexpr double          MIN_SCORE = -std::numeric_limits<double>::max();
        std::array<MoveResult, 4> moves{
            MoveResult{label.vio}, // No operation
            MoveResult{},          // Add
            MoveResult{},          // Remove
            MoveResult{}           // Swap
        };

        // Determine which operations to perform based on search direction
        const bool can_add    = label.search_dir == 'a' || label.search_dir == 's';
        const bool can_remove = label.search_dir == 'r' || label.search_dir == 's';
        const bool can_swap   = label.search_dir == 'a' || label.search_dir == 'r';

        double new_vio = MIN_SCORE;

        // Perform valid operations
        if (can_add) {
            int add_j;
            addSearchCrazy(label.plan_idx, label.c, label.w_no_c, new_vio, add_j);
            moves[1]                = MoveResult{new_vio};
            moves[1].operation_data = add_j;
        }

        if (can_remove) {
            int remove_j;
            removeSearchCrazy(label.plan_idx, label.c, new_vio, remove_j);
            moves[2]                = MoveResult{new_vio};
            moves[2].operation_data = remove_j;
        }

        if (can_swap) {
            std::pair<int, int> swap_pair;
            swapSearchCrazy(label.plan_idx, label.c, label.w_no_c, new_vio, swap_pair);
            moves[3]                = MoveResult{new_vio};
            moves[3].operation_data = swap_pair;
        }

        // Find best move
        const auto   best_move_it = std::max_element(moves.begin(), moves.end(), [](const auto &a, const auto &b) {
            return a.violation_score < b.violation_score;
        });
        const size_t best_idx     = std::distance(moves.begin(), best_move_it);
        const auto  &best_move    = *best_move_it;

        // Handle no operation case
        if (best_idx == 0) {
            cutLong tmp;
            for (const int j : label.c) { tmp.set(j); }
            generated_rank1_multi_pool[label.c.size()].emplace_back(tmp, label.plan_idx, best_move.violation_score);
            ++i;
            return;
        }

        // Execute the chosen operation using std::visit
        std::visit(overloaded{[](const std::monostate &) { /* No operation */ },
                              [&](const int j) {
                                  if (best_idx == 1) { // Add
                                      label.w_no_c.erase(std::remove(label.w_no_c.begin(), label.w_no_c.end(), j),
                                                         label.w_no_c.end());
                                      label.c.push_back(j);
                                  } else { // Remove
                                      label.c.erase(std::remove(label.c.begin(), label.c.end(), j), label.c.end());
                                  }
                              },
                              [&](const std::pair<int, int> &swap_pair) {
                                  const auto [from, to] = swap_pair;
                                  if (auto it = std::find(label.c.begin(), label.c.end(), from); it != label.c.end()) {
                                      *it = to;
                                      label.w_no_c.erase(std::remove(label.w_no_c.begin(), label.w_no_c.end(), to),
                                                         label.w_no_c.end());
                                  }
                              }},
                   best_move.operation_data);

        label.vio = best_move.violation_score;
    }

    // Helper struct for std::visit
    template <class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    void chooseCuts(const std::vector<R1c> &tmp_cuts, std::vector<R1c> &chosen_cuts, int numCuts) {
        numCuts = std::min(numCuts, static_cast<int>(tmp_cuts.size()));
        if (numCuts == 0) return;

        chosen_cuts.reserve(numCuts); // Reserve space for the chosen cuts to avoid repeated reallocations

        for (const auto &cut : tmp_cuts) {
            const auto &fst = cut.info_r1c.first;
            const auto &snd = cut.info_r1c.second;

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

    struct State {
        int              begin;
        int              target_remainder;
        std::vector<int> remaining_counts;
        std::vector<int> memory_indices;

        State(int b, int t, std::vector<int> rc, std::vector<int> mi)
            : begin(b), target_remainder(t), remaining_counts(std::move(rc)), memory_indices(std::move(mi)) {}
    };

    void findPlanForRank1Multi(const std::vector<int> &vis, const int denominator, cutLong &mem,
                               std::vector<ankerl::unordered_dense::set<int>> &segment,
                               std::vector<std::vector<int>>                  &plan) {
        // Calculate initial sum and remainder
        const int sum         = std::accumulate(vis.begin(), vis.end(), 0);
        const int initial_mod = sum % denominator;

        // Create key for memoization
        auto key = vis;
        key.push_back(initial_mod);

        auto &cached_result = rank1_multi_mem_plan_map[key];

        // Generate solutions if not cached
        if (cached_result.empty()) {
            std::deque<State> states;
            states.emplace_back(0, initial_mod, vis, std::vector<int>{});

            while (!states.empty()) {
                auto [begin, target_rem, remaining, memory] = std::move(states.front());
                states.pop_front();

                int running_sum = 0;
                for (size_t j = 0; j < remaining.size(); ++j) {
                    running_sum = (running_sum + remaining[j]) % denominator;

                    if (running_sum > 0) {
                        const size_t current_pos = begin + j;

                        // Try branching if possible
                        if (running_sum <= target_rem && j + 1 < remaining.size()) {
                            std::vector<int> new_remaining(remaining.begin() + j + 1, remaining.end());
                            states.emplace_back(current_pos + 1, target_rem - running_sum, std::move(new_remaining),
                                                memory);
                        }

                        // Add to memory if not last element
                        if (current_pos != vis.size() - 1) { memory.push_back(current_pos); }
                    }
                }
                cached_result.push_back(std::move(memory));
            }
        }

        // Process memory constraints
        for (int i = 1; i < dim; ++i) {
            if (mem[i]) {
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        // Build visibility map
        std::vector<ankerl::unordered_dense::set<int>> vertex_visibility(dim);
        for (size_t plan_idx = 0; plan_idx < cached_result.size(); ++plan_idx) {
            for (int j : cached_result[plan_idx]) {
                for (int k : segment[j]) { vertex_visibility[k].insert(plan_idx); }
            }
        }

        // Update memory based on visibility
        for (int i = 1; i < dim; ++i) {
            if (vertex_visibility[i].size() == cached_result.size()) {
                mem.set(i);
                for (auto &seg : segment) { seg.erase(i); }
            }
        }

        // Process memory patterns
        std::vector<std::pair<cutLong, std::vector<int>>> memory_patterns;
        memory_patterns.reserve(cached_result.size());

        for (const auto &memory_indices : cached_result) {
            cutLong pattern_mem;

            // Build memory pattern
            for (int j : memory_indices) {
                for (int k : segment[j]) { pattern_mem.set(k); }
            }

            // Check for empty pattern
            if (pattern_mem.none()) {
                plan.clear();
                return;
            }

            // Find or add pattern
            auto it = std::find_if(memory_patterns.begin(), memory_patterns.end(), [&pattern_mem](const auto &entry) {
                return ((pattern_mem & entry.first) ^ pattern_mem).none();
            });

            if (it != memory_patterns.end()) {
                it->first  = std::move(pattern_mem);
                it->second = memory_indices;
            } else {
                memory_patterns.emplace_back(std::move(pattern_mem), memory_indices);
            }
        }

        // Build final plan
        plan.resize(memory_patterns.size());
        std::transform(memory_patterns.begin(), memory_patterns.end(), plan.begin(),
                       [](const auto &entry) { return entry.second; });
    }

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

    // New members for R1C separation
    struct R1CMemory {
        std::vector<int> C;          // Customer subset
        int              p_idx;      // Index of optimal vector p
        yzzLong          arc_memory; // Arc memory as bitset
        double           violation;  // Cut violation

        R1CMemory(std::vector<int> c, int p, double v) : C(std::move(c)), p_idx(p), violation(v) {}
    };

private:
};
