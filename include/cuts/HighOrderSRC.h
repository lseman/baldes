/*
 * @file HighDimCutsGenerator.h
 * @brief High-dimensional cuts generator for the VRP
 *
 * This class is responsible for generating high-dimensional cuts for the VRP.
 * The class is based on RouteOpt: https://github.com/Zhengzhong-You/RouteOpt/
 *
 */

#pragma once

#include <exec/static_thread_pool.hpp>
#include <ranges>
#include <stdexec/execution.hpp>

#include "Bitset.h"
#include "Common.h"
#include "Path.h"
#include "RCC.h"
#include "SparseMatrix.h"
#include "VRPNode.h"
#include "utils/Hashes.h"

// include intel concurrent hashmap
// #include "ConcurrentMap.h"
#include <libcuckoo/cuckoohash_map.hh>

#include "HighHelper.h"

constexpr int INITIAL_RANK_1_MULTI_LABEL_POOL_SIZE = 50;
constexpr int INITIAL_POOL_SIZE = 100;
constexpr double tolerance = 1e-3;
constexpr int max_row_rank1 = 5;
constexpr int max_heuristic_initial_seed_set_size_row_rank1c = 6;

constexpr int max_num_r1c_per_round = 2;
constexpr double cut_vio_factor = 0.1;

class HighDimCutsGenerator {
   public:
    std::future<void> backgroundTask;  // Future to track the background task

    std::shared_ptr<HighDimCutsGenerator> cloned;

    std::shared_ptr<HighDimCutsGenerator> clone() const {
        // Directly create a new instance without copying `pool_mutex`
        auto new_gen = std::shared_ptr<HighDimCutsGenerator>(
            new HighDimCutsGenerator(N_SIZE));
        new_gen->cost_mat4_vertex = this->cost_mat4_vertex;
        new_gen->nodes = this->nodes;
        new_gen->arc_duals = this->arc_duals;
        new_gen->max_heuristic_sep_mem4_row_rank1 = 16;
        new_gen->initialize(this->sol);
        new_gen->generateSepHeurMem4Vertex();

        return new_gen;
    }

    void runTask() {
        getHighDimCuts();
        setMemFactor(0.75);
        constructMemoryVertexBased();
    }

    // Start generating cuts in the background
    void generateCutsInBackground() {
        if (backgroundTask.valid()) {
            return;
        }
        // print_info("Generating cuts in the background\n");
        // cloned = this->clone();
        backgroundTask =
            std::async(std::launch::async, [this]() { runTask(); });
    }

    std::vector<R1c> returnBGcuts() { return getCuts(); }

    bool checkBackgroundTask() {
        if (backgroundTask.valid()) {
            return true;
        }
        return false;
    }

    // Retrieve generated cuts
    bool readyGeneratedCuts() {
        if (!backgroundTask.valid()) {
            return false;
        }

        // Check if the background task has completed
        if (backgroundTask.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
            backgroundTask = std::future<void>();  // Reset the future
            return true;
        } else {
            return false;
        }
    }

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

    HighDimCutsGenerator(int dim) : dim(dim), num_label(0) {
        cuts.reserve(INITIAL_POOL_SIZE);
        rank1_multi_label_pool.reserve(INITIAL_POOL_SIZE);
        generated_rank1_multi_pool.reserve(INITIAL_POOL_SIZE);
        // map_cut_plan_vio.reserve(INITIAL_POOL_SIZE);

        generateOptimalMultiplier();
        // printMultiplierMap();
    }

    void setMemFactor(double factor) { max_cut_mem_factor = factor; }

    void generateOptimalMultiplier();
    void printCuts();

    std::vector<R1c> &getCuts() { return cuts; }

    ankerl::unordered_dense::map<
        int, std::vector<std::tuple<std::vector<int>, int, int>>>
        map_rank1_multiplier;
    ankerl::unordered_dense::map<int, ankerl::unordered_dense::set<int>>
        map_rank1_multiplier_dominance{};
    std::vector<std::vector<std::vector<std::vector<int>>>>
        record_map_rank1_combinations{};

    void printMultiplierMap();

    void generatePermutations(ankerl::unordered_dense::map<int, int> &count_map,
                              std::vector<int> &result,
                              std::vector<std::vector<int>> &results,
                              int remaining);

    SparseMatrix matrix;
    void initialize(const std::vector<Path> &routes) {
        this->sol = routes;
        initialSupportvector();
        cacheValidRoutesAndMappings();  // Call new caching function
    }

    std::vector<VRPNode> nodes;
    void setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    std::vector<Path> sol;
    int dim, max_row_rank1, num_label;
    // std::vector<ankerl::unordered_dense::map<int, int>>        v_r_map;
    gch::small_vector<std::vector<int>> v_r_map;
    gch::small_vector<std::pair<std::vector<int>, std::vector<int>>> c_N_noC;
    std::vector<Rank1MultiLabel> rank1_multi_label_pool;
    ankerl::unordered_dense::map<
        int, std::vector<std::tuple<Bitset<N_SIZE>, int, double>>>
        generated_rank1_multi_pool;
    std::vector<R1c> cuts;
    std::vector<R1c> old_cuts;

    ankerl::unordered_dense::map<std::vector<int>,
                                 std::vector<std::vector<int>>, VectorHash>
        rank1_multi_mem_plan_map;
    ankerl::unordered_dense::map<std::vector<int>,
                                 ankerl::unordered_dense::set<int>, VectorHash>
        cut_record;
    std::vector<std::pair<int, double>> move_vio;
    std::vector<yzzLong> rank1_sep_heur_mem4_vertex;

    void initialSupportvector() {
        cut_record.clear();
        v_r_map.clear();
        c_N_noC.clear();
        map_cut_plan_vio.clear();
        // map_cut_plan_vio.reserve(4096);
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
        // Step 1: Initialize data structures and prepare for processing
        constructVRMapAndSeedCrazy();
        startSeedCrazy();

        // Step 2: Set up parallel processing
        std::atomic<int> atomic_i{
            0};  // Atomic counter for thread-safe index tracking
        const int JOBS =
            std::thread::hardware_concurrency();  // Number of threads
        exec::static_thread_pool pool(JOBS);      // Create a thread pool
        auto sched =
            pool.get_scheduler();  // Get the scheduler for the thread pool

        // Step 3: Define the parallel task
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), JOBS, [this, &atomic_i](std::size_t) {
                while (true) {
                    // Atomically fetch the next index to process
                    int current_i = atomic_i.fetch_add(1);

                    // Exit if all labels have been processed
                    if (current_i >= num_label) {
                        break;
                    }

                    // Process the current label
                    operationsCrazy(rank1_multi_label_pool[current_i],
                                    current_i);
                }
            });

        // Step 4: Submit the parallel task to the thread pool and wait for
        // completion
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Step 5: Construct the final cuts and update the old cuts
        constructCutsCrazy();
        old_cuts = cuts;
    }

    // declare rank1_mutex
    std::mutex rank1_mutex;

    void startSeedCrazy() {
        num_label = 0;

        // Create thread pool
        const int JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS);
        auto sched = pool.get_scheduler();

        // Preallocate work items
        std::vector<std::pair<int, std::reference_wrapper<const std::pair<
                                       std::vector<int>, std::vector<int>>>>>
            work_items;
        work_items.reserve(7 * c_N_noC.size());

        // Populate work items
        for (int plan_idx : {0, 1, 2, 3, 4, 5, 6}) {
            for (const auto &pair : c_N_noC) {
                work_items.emplace_back(plan_idx, std::ref(pair));
            }
        }

        // Define chunk size to reduce parallelization overhead
        const int chunk_size =
            std::max(1, static_cast<int>(work_items.size()) / JOBS);

        // Thread-local storage for results to reduce lock contention
        struct ThreadLocalResult {
            std::vector<Rank1MultiLabel> labels;
            std::mutex mutex;
        };
        std::vector<ThreadLocalResult> thread_local_results(JOBS);

        // Parallel processing using bulk sender
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (work_items.size() + chunk_size - 1) / chunk_size,
            [this, &work_items, chunk_size, JOBS,
             &thread_local_results](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx =
                    std::min(start_idx + chunk_size, work_items.size());

                // Process a chunk of tasks
                for (size_t task_idx = start_idx; task_idx < end_idx;
                     ++task_idx) {
                    const auto &[plan_idx, pair_ref] = work_items[task_idx];
                    const auto &[c, wc] = pair_ref.get();

                    if (c.empty() || wc.empty()) {
                        continue;
                    }

                    std::vector<int> c_mutable = c;
                    double initial_vio = 0.0;

                    exactFindBestPermutationForOnePlan(c_mutable, plan_idx,
                                                       initial_vio);

                    if (initial_vio < tolerance) {
                        continue;
                    }

                    // Additional operations: add, remove, swap
                    int add_j, remove_j;
                    double add_vio = initial_vio, remove_vio = initial_vio;
                    std::pair<int, int> swap_i_j;
                    double swap_vio = initial_vio;

                    addSearchCrazy(plan_idx, c_mutable, wc, add_vio, add_j);
                    removeSearchCrazy(plan_idx, c_mutable, remove_vio,
                                      remove_j);
                    swapSearchCrazy(plan_idx, c_mutable, wc, swap_vio,
                                    swap_i_j);

                    double best_vio = std::max({add_vio, remove_vio, swap_vio});
                    if (best_vio > initial_vio) {
                        std::vector<int> new_c, new_w_no_c;
                        char best_op = 'n';

                        if (best_vio == add_vio) {
                            best_op = 'a';
                            applyBestOperation('a', c_mutable, wc, add_j, -1,
                                               std::make_pair(-1, -1), new_c,
                                               new_w_no_c, plan_idx, best_vio);
                        } else if (best_vio == remove_vio) {
                            best_op = 'r';
                            applyBestOperation('r', c_mutable, wc, -1, remove_j,
                                               std::make_pair(-1, -1), new_c,
                                               new_w_no_c, plan_idx, best_vio);
                        } else if (best_vio == swap_vio) {
                            best_op = 's';
                            applyBestOperation('s', c_mutable, wc, -1, -1,
                                               swap_i_j, new_c, new_w_no_c,
                                               plan_idx, best_vio);
                        }

                        // Store results in thread-local storage
                        auto &local_result =
                            thread_local_results[chunk_idx % JOBS];
                        std::lock_guard<std::mutex> lock(local_result.mutex);
                        local_result.labels.emplace_back(
                            std::move(new_c), std::move(new_w_no_c), plan_idx,
                            best_vio, best_op);
                    }
                }
            });

        // Submit work to the thread pool and wait for completion
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Merge thread-local results into the global pool
        for (auto &local_result : thread_local_results) {
            std::lock_guard<std::mutex> lock(
                rank1_mutex);  // Fixed: Use rank1_mutex
            if (num_label + local_result.labels.size() >
                rank1_multi_label_pool.size()) {
                rank1_multi_label_pool.resize(std::max<std::size_t>(
                    rank1_multi_label_pool.size() * 2,
                    num_label + local_result.labels.size()));
            }
            for (auto &label : local_result.labels) {
                rank1_multi_label_pool[num_label++] = std::move(label);
            }
        }
    }

    void applyBestOperation(char best_oper, const std::vector<int> &c,
                            const std::vector<int> &wc, int add_j, int remove_j,
                            const std::pair<int, int> &swap_i_j,
                            std::vector<int> &new_c,
                            std::vector<int> &new_w_no_c, int plan_idx,
                            double best_vio) {
        switch (best_oper) {
            case 'o':  // No operation, directly add to target pool
                if (c.size() >= 2 && c.size() <= max_row_rank1) {
                    cutLong tmp;
                    for (int i : c) tmp.set(i);
                    {
                        std::lock_guard<std::mutex> lock(pool_mutex);
                        generated_rank1_multi_pool[static_cast<int>(c.size())]
                            .emplace_back(tmp, plan_idx, best_vio);
                    }
                }
                break;

            case 'a':  // Add operation
                new_c = c;
                new_c.push_back(add_j);
                new_w_no_c.reserve(wc.size() - 1);
                std::copy_if(wc.begin(), wc.end(),
                             std::back_inserter(new_w_no_c),
                             [add_j](int w) { return w != add_j; });
                break;

            case 'r':  // Remove operation
                new_c.reserve(c.size() - 1);
                std::copy_if(c.begin(), c.end(), std::back_inserter(new_c),
                             [remove_j](int i) { return i != remove_j; });
                new_w_no_c = wc;
                break;

            case 's':  // Swap operation
                new_c = c;
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

    std::mutex pool_mutex;  // Mutex to protect shared pool access

    gch::small_vector<double> cached_valid_routes;
    gch::small_vector<int> cached_map_old_new_routes;
    int num_valid_routes;

    void cacheValidRoutesAndMappings() {
        cached_valid_routes.clear();
        cached_map_old_new_routes.clear();

        cached_valid_routes.reserve(sol.size());
        cached_map_old_new_routes.resize(sol.size(), -1);

        num_valid_routes = 0;

        std::for_each(sol.begin(), sol.end(),
                      [this, current = 0](const auto &route) mutable {
                          if (route.frac_x > 1e-2) {
                              cached_valid_routes.push_back(route.frac_x);
                              cached_map_old_new_routes[current] =
                                  num_valid_routes++;
                          }
                          ++current;
                      });
    }

    // cutLong as key
    libcuckoo::cuckoohash_map<cutLong,
                              std::vector<std::pair<std::vector<int>, double>>>
        map_cut_plan_vio;

    // cut_cache uses vector<int> as key
    libcuckoo::cuckoohash_map<std::vector<int>, std::vector<std::vector<int>>,
                              VectorIntHashCompare>
        cut_cache;

    using KeyType = Bitset<N_SIZE>;
    using ValueType = std::vector<std::pair<std::vector<int>, double>>;
    // ankerl::unordered_dense::map<Bitset<N_SIZE>,
    // std::vector<std::pair<std::vector<int>, double>>> map_cut_plan_vio;

    // declare map_cut_plan_vio_mutex
    std::shared_mutex map_cut_plan_vio_mutex;
    std::shared_mutex cut_cache_mutex;

    // For the frequency accumulation loop:
    void accumulate_frequencies_simd(int *__restrict__ map_r_numbers,
                                     const int *__restrict__ route_freqs,
                                     const int route_size) noexcept {
        static constexpr int SIMD_SIZE =
            8;  // AVX2 processes 8 integers at once
        static constexpr int PREFETCH_DISTANCE = 64;  // Common cache line size

        // Ensure proper alignment for best performance
        alignas(32) const int *freq_ptr = route_freqs;
        alignas(32) int *map_ptr = map_r_numbers;

        int i = 0;

        // Process 8 elements at a time using AVX2
        for (; i + SIMD_SIZE <= route_size; i += SIMD_SIZE) {
            // Prefetch next cache lines
            _mm_prefetch(reinterpret_cast<const char *>(freq_ptr + i +
                                                        PREFETCH_DISTANCE),
                         _MM_HINT_T0);
            _mm_prefetch(
                reinterpret_cast<const char *>(map_ptr + i + PREFETCH_DISTANCE),
                _MM_HINT_T0);

            // Load frequencies and current values
            __m256i freq = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(&freq_ptr[i]));
            __m256i curr = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(&map_ptr[i]));

            // Mask frequencies greater than 0
            __m256i zero = _mm256_setzero_si256();
            __m256i mask = _mm256_cmpgt_epi32(freq, zero);
            freq = _mm256_and_si256(freq, mask);

            // Accumulate
            curr = _mm256_add_epi32(curr, freq);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&map_ptr[i]), curr);
        }

        // Handle remaining elements with auto-vectorization hint
#pragma omp simd
        for (; i < route_size; i++) {
            const int frequency = freq_ptr[i];
            map_ptr[i] += frequency * (frequency > 0);
        }
    }

    struct ViolationResult {
        double violation;
        int index;
        bool operator<(const ViolationResult &other) const {
            return violation < other.violation;
        }
        bool operator>(const ViolationResult &other) const {
            return violation > other.violation;
        }
        // operator for max element
        bool operator<(const double &other) const { return violation < other; }
    };

    void exactFindBestPermutationForOnePlan(std::vector<int> &cut,
                                            const int plan_idx, double &vio) {
        const int cut_size = static_cast<int>(cut.size());
        const auto &plan = map_rank1_multiplier[cut_size][plan_idx];

        // Early exit if the plan denominator is zero
        if (get<1>(plan) == 0) {
            vio = -std::numeric_limits<double>::max();
            return;
        }

        // Thread-local storage with aligned vectors
        alignas(64) static thread_local cutLong tmp;
        alignas(64) static thread_local std::vector<int, aligned_allocator<int>>
            map_r_numbers;
        alignas(64) static thread_local std::vector<double>
            num_times_vis_routes;
        alignas(64) static thread_local std::vector<std::vector<int>>
            cut_num_times_vis_routes;
        alignas(64) static thread_local std::vector<std::pair<int, int>>
            cut_coeff;
        alignas(64) static thread_local std::vector<int> new_cut;

        // Reset the temporary cutLong
        tmp.reset();

        // Filter valid indices and set them in the temporary cutLong
        for (int it : cut) {
            if (it >= 0 && static_cast<size_t>(it) < v_r_map.size()) {
                tmp.set(it);
            }
        }

        // Check if the result is already cached
        std::vector<std::pair<std::vector<int>, double>> cached_plans;
        if (map_cut_plan_vio.find(tmp, cached_plans) &&
            plan_idx < cached_plans.size() &&
            !cached_plans[plan_idx].first.empty()) {
            vio = cached_plans[plan_idx].second;
            return;
        }

        // Insert a new entry into the cache if not found
        map_cut_plan_vio.insert(
            tmp, std::vector<std::pair<std::vector<int>, double>>(7));

        // Extract plan details
        const int denominator = get<1>(plan);
        const double rhs = get<2>(plan);
        const auto &coeffs = record_map_rank1_combinations[cut_size][plan_idx];

        // Initialize map_r_numbers with zeros
        map_r_numbers.assign(sol.size(), 0);

        // Process frequencies using SIMD (if AVX2 is available) or scalar
        // operations
#ifdef __AVX2__
        for (int i = 0; i < cut_size; ++i) {
            const int idx = cut[i];
            if (idx >= 0 && static_cast<size_t>(idx) < v_r_map.size()) {
                const auto &freqs = v_r_map[idx];
                accumulate_frequencies_simd(map_r_numbers.data(), freqs.data(),
                                            static_cast<int>(freqs.size()));
            }
        }
#else
        for (int i = 0; i < cut_size; ++i) {
            const int idx = cut[i];
            if (idx >= 0 && static_cast<size_t>(idx) < v_r_map.size()) {
                const auto &freqs = v_r_map[idx];
                for (size_t j = 0; j < freqs.size(); ++j) {
                    const int freq = freqs[j];
                    map_r_numbers[j] += freq * (freq > 0);
                }
            }
        }
#endif

        // Handle cache for cut_num_times_vis_routes
        std::vector<std::vector<int>> cached_routes;
        if (cut_cache.find(cut, cached_routes)) {
            cut_num_times_vis_routes = cached_routes;
        } else {
            cut_num_times_vis_routes.clear();
            cut_num_times_vis_routes.resize(cut_size);

            // Reserve space for each route
            for (auto &vec : cut_num_times_vis_routes) {
                vec.reserve(num_valid_routes * 2);
            }

            // Populate cut_num_times_vis_routes
            for (int i = 0; i < cut_size; ++i) {
                const int c = cut[i];
                if (c >= 0 && static_cast<size_t>(c) < v_r_map.size()) {
                    auto &current_route = cut_num_times_vis_routes[i];
                    const auto &route_freqs = v_r_map[c];

                    for (size_t route_idx = 0; route_idx < route_freqs.size();
                         ++route_idx) {
                        const int frequency = route_freqs[route_idx];
                        const int cached_route =
                            cached_map_old_new_routes[route_idx];
                        if (frequency > 0 && cached_route != -1) {
                            current_route.insert(current_route.end(), frequency,
                                                 cached_route);
                        }
                    }
                }
            }

            // Insert into cache
            cut_cache.insert(cut, cut_num_times_vis_routes);
        }

        // Process violations
        num_times_vis_routes.resize(num_valid_routes);
        std::vector<ViolationResult> violations;
        violations.reserve(coeffs.size());

        for (size_t coeff_idx = 0; coeff_idx < coeffs.size(); ++coeff_idx) {
            const auto &coeff_row = coeffs[coeff_idx];

            // Use stack-allocated array for small sizes, heap otherwise
            constexpr size_t STACK_SIZE = 1024;
            std::array<double, STACK_SIZE> local_times{};
            std::unique_ptr<double[]> heap_times;
            std::span<double> times_span;

            if (num_valid_routes <= STACK_SIZE) {
                times_span = std::span{local_times}.first(num_valid_routes);
            } else {
                heap_times = std::make_unique<double[]>(num_valid_routes);
                times_span = std::span{heap_times.get(),
                                       static_cast<size_t>(num_valid_routes)};
            }
            std::ranges::fill(times_span, 0.0);

            // Accumulate times for each route
            for (size_t i = 0; i < cut_num_times_vis_routes.size(); ++i) {
                const double coeff = coeff_row[i];
                const auto &routes = cut_num_times_vis_routes[i];
                for (const int route : routes) {
                    times_span[route] += coeff;
                }
            }

            // Calculate violation
            double vio_tmp = -rhs;
            for (size_t i = 0; i < num_valid_routes; ++i) {
                vio_tmp +=
                    static_cast<int>(times_span[i] / denominator + tolerance) *
                    cached_valid_routes[i];
            }

            violations.emplace_back(vio_tmp, static_cast<int>(coeff_idx));
        }

        // Find the best violation
        const auto best_result = *std::ranges::max_element(
            violations, [](const ViolationResult &a, const ViolationResult &b) {
                return a.violation < b.violation;
            });
        vio = best_result.violation;

        // Prepare cut_coeff and new_cut
        cut_coeff.resize(cut_size);
        if (static_cast<size_t>(best_result.index) < coeffs.size()) {
            const auto &best_coeffs = coeffs[best_result.index];

            for (int i = 0; i < cut_size; ++i) {
                cut_coeff[i] = {cut[i], i < static_cast<int>(best_coeffs.size())
                                            ? static_cast<int>(best_coeffs[i])
                                            : 0};
            }
        }

        // Sort cut_coeff in descending order of coefficients
        pdqsort(cut_coeff.begin(), cut_coeff.end(),
                [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                    return a.second > b.second;
                });

        // Create new_cut from sorted cut_coeff
        new_cut.resize(cut_size);
        std::ranges::transform(cut_coeff, new_cut.begin(),
                               &std::pair<int, int>::first);

        // Update the cache with the new cut and violation
        map_cut_plan_vio.upsert(tmp, [&](auto &value) {
            value.resize(
                std::max(value.size(), static_cast<size_t>(plan_idx + 1)));
            value[plan_idx] = {new_cut, vio};
            return true;
        });
    }

    std::vector<std::vector<double>> cost_mat4_vertex;

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) {
        cost_mat4_vertex = distances;
    }

#if defined(RCC) || defined(EXACT_RCC)
    ArcDuals arc_duals;
    void setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif

    void generateSepHeurMem4Vertex() {
        // Early exit for invalid dimensions
        if (dim <= 0) return;

        // Resize and precompute half costs
        rank1_sep_heur_mem4_vertex.resize(dim);
        std::vector<double> half_cost(dim);
        for (int i = 0; i < dim; ++i) {
            half_cost[i] = nodes[i].cost / 2;
        }

        // Determine the number of clusters (k) for k-means
        const int k = std::min(5, dim / 2);  // Ensure at least 1 cluster

        // Initialize clusters and assignments
        std::vector<Cluster> clusters(k);
        std::vector<int> assignments(dim);
        for (int i = 0; i < dim; ++i) {
            int cluster = i % k;  // Simple initial assignment (can be improved)
            clusters[cluster].vertices.push_back(i);
            assignments[i] = cluster;
        }

        // Iterate through each vertex to generate candidates
        for (int i = 0; i < dim; ++i) {
            std::vector<std::pair<int, double>> candidates;
            candidates.reserve(dim);

            // Generate candidates from all clusters
            for (const auto &cluster : clusters) {
                for (int j : cluster.vertices) {
                    if (i != j) {
                        // Compute the cost for the candidate
                        double cost = cost_mat4_vertex[i][j] -
                                      (half_cost[i] + half_cost[j]) -
                                      arc_duals.getDual(i, j);
                        candidates.emplace_back(j, cost);
                    }
                }
            }

            // Sort candidates by cost in ascending order
            pdqsort(candidates.begin(), candidates.end(),
                    [](const auto &a, const auto &b) {
                        return a.second < b.second;
                    });

            // Select top candidates ensuring cluster diversity
            cutLong &vst2 = rank1_sep_heur_mem4_vertex[i];
            vst2.reset();

            ankerl::unordered_dense::set<int>
                used_clusters;  // Tracks clusters already used
            int selected = 0;

            for (const auto &candidate : candidates) {
                int cluster = assignments[candidate.first];
                if (used_clusters.insert(cluster).second ||
                    used_clusters.size() == k) {
                    vst2.set(candidate.first);
                    if (++selected == max_heuristic_sep_mem4_row_rank1) break;
                }
            }
        }
    }

    void constructVRMapAndSeedCrazy() {
        // Resize `rank1_sep_heur_mem4_vertex` and initialize `v_r_map`
        rank1_sep_heur_mem4_vertex.resize(dim);
        v_r_map.assign(dim, std::vector<int>(sol.size(), 0));
        const int max_dim_minus_1 = dim - 1;
        const auto sol_size = sol.size();

        // Populate `v_r_map` with route appearances for each vertex `i`
#ifdef __cpp_lib_parallel_algorithm
        std::for_each(sol.begin(), sol.end(), [&](const auto &route_data) {
            const auto r =
                &route_data - &sol[0];  // Get index without separate counter
            for (const int i : route_data.route) {
                if (i > 0 && i < max_dim_minus_1) {
                    ++v_r_map[i][r];
                }
            }
        });
#else
        for (size_t r = 0; r < sol_size; ++r) {
            for (const int i : sol[r].route) {
                if (i > 0 && i < max_dim_minus_1) {
                    ++v_r_map[i][r];
                }
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
            cutLong working_set;
            const auto &heur_mem = rank1_sep_heur_mem4_vertex[i];

            // Build working set from routes containing vertex i
            for (size_t route_idx = 0; route_idx < v_r_map[i].size();
                 ++route_idx) {
                if (v_r_map[i][route_idx] <= 0) continue;

                for (const int v : sol[route_idx].route) {
                    if (v > 0 && v < max_dim_minus_1 && heur_mem.test(v)) {
                        working_set.set(v);
                    }
                }
            }

            // Process routes not containing vertex i
            for (size_t r = 0; r < sol_size; ++r) {
                if (v_r_map[i][r] > 0) continue;

                cutLong candidate_set;
                for (const int v : sol[r].route) {
                    if (v > 0 && v < max_dim_minus_1 && working_set.test(v)) {
                        candidate_set.set(v);
                    }
                }
                candidate_set.set(i);

                const auto c_size = candidate_set.count();
                if (c_size >= 4 &&
                    c_size <= max_heuristic_initial_seed_set_size_row_rank1c) {
                    const auto complement = working_set ^ candidate_set;
                    auto [it, inserted] =
                        seed_map.try_emplace(candidate_set, complement);
                    if (!inserted) {
                        it->second |= complement;
                    }
                }
            }
        }

        // Prepare `c_N_noC` by transforming `seed_map` entries
        c_N_noC.resize(seed_map.size());
        std::transform(seed_map.begin(), seed_map.end(), c_N_noC.begin(),
                       [max_dim_minus_1](const auto &seed_pair) {
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
    // Helper for finding best violation
    template <typename F>
    static inline std::pair<double, int> findBestViolation(
        F &&calculate_violation, const std::vector<int> &candidates,
        double initial_vio = -std::numeric_limits<double>::max()) noexcept {
        // Early return for empty candidates
        if (candidates.empty()) {
            return {initial_vio, -1};
        }

        double best_vio = initial_vio;
        int best_candidate = -1;

        for (const int candidate : candidates) {
            const double vio = calculate_violation(candidate);
            if (vio > best_vio) {
                best_vio = vio;
                best_candidate = candidate;
            }
        }

        return {best_vio, best_candidate};
    }

    inline void shiftSearchCrazy(int plan_idx, const std::vector<int> &c,
                                 double &new_vio,
                                 std::pair<int, int> &shift_positions) {
        // Validate size and plan
        auto validation = PlanValidationResult::validate(plan_idx, c.size(),
                                                         map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        std::vector<int> tmp_c = c;
        double best_vio = -std::numeric_limits<double>::max();
        std::pair<int, int> best_shift{-1, -1};

        // Try shifting elements to different positions
        for (int from = 0; from < c.size(); ++from) {
            const int element = c[from];

            // Find best position to shift this element to
            auto [pos_best_vio, to_pos] = findBestViolation(
                [&](int to) {
                    if (to == from) return -std::numeric_limits<double>::max();

                    // Perform the shift
                    if (to < from) {
                        // Shift up - move element earlier in sequence
                        std::rotate(tmp_c.begin() + to, tmp_c.begin() + from,
                                    tmp_c.begin() + from + 1);
                    } else {
                        // Shift down - move element later in sequence
                        std::rotate(tmp_c.begin() + from,
                                    tmp_c.begin() + from + 1,
                                    tmp_c.begin() + to + 1);
                    }

                    double vio;
                    // fmt::print("Calling in shiftSearchCrazy\n");
                    exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

                    // Restore original order
                    tmp_c = c;
                    return vio;
                },
                std::vector<int>(c.size()));  // Try all possible positions

            if (pos_best_vio > best_vio) {
                best_vio = pos_best_vio;
                best_shift = {from, to_pos};
            }
        }

        new_vio = best_vio;
        shift_positions = best_shift;
    }

    inline void addSearchCrazy(int plan_idx, const std::vector<int> &c,
                               const std::vector<int> &w_no_c, double &new_vio,
                               int &add_j) {
        const int new_size = c.size() + 1;

        // Validate size and plan
        if (new_size > max_row_rank1) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        auto validation = PlanValidationResult::validate(plan_idx, new_size,
                                                         map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        // Prepare temporary vector with space for the new candidate
        std::vector<int> tmp_c = c;
        tmp_c.push_back(0);  // Placeholder for the candidate

        double best_vio = -std::numeric_limits<double>::max();
        int best_candidate = -1;

        // Find the best candidate
        for (const int candidate : w_no_c) {
            tmp_c.back() = candidate;

            // Compute violation
            double vio;
            // fmt::print("Calling in addSearchCrazy\n");
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

            if (vio > best_vio) {
                best_vio = vio;
                best_candidate = candidate;
            }
        }

        new_vio = best_vio - tolerance;
        add_j = best_candidate;
    }

    inline void removeSearchCrazy(int plan_idx, const std::vector<int> &c,
                                  double &new_vio, int &remove_j) {
        const int new_size = c.size() - 1;

        // Validate size and plan
        auto validation = PlanValidationResult::validate(plan_idx, new_size,
                                                         map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = -std::numeric_limits<double>::max();
            return;
        }

        // Prepare temporary vector for reduced size
        std::vector<int> tmp_c(new_size);

        double best_vio = -std::numeric_limits<double>::max();
        int best_idx = -1;

        // Iterate over all indices to find the best removal
        for (int idx = 0; idx < static_cast<int>(c.size()); ++idx) {
            // Create the reduced vector by skipping the current index
            std::copy(c.begin(), c.begin() + idx, tmp_c.begin());
            std::copy(c.begin() + idx + 1, c.end(), tmp_c.begin() + idx);

            // Compute violation
            double vio;
            // fmt::print("Calling in removeSearchCrazy\n");
            exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

            if (vio > best_vio) {
                best_vio = vio;
                best_idx = idx;
            }
        }

        new_vio = best_vio + tolerance;
        remove_j = best_idx >= 0 ? c[best_idx] : -1;
    }

    inline void swapSearchCrazy(int plan_idx, const std::vector<int> &c,
                                const std::vector<int> &w_no_c, double &new_vio,
                                std::pair<int, int> &swap_i_j) {
        constexpr double MIN_SCORE = -std::numeric_limits<double>::max();

        // Early validation check
        auto validation = PlanValidationResult::validate(plan_idx, c.size(),
                                                         map_rank1_multiplier);
        if (!validation.is_valid) {
            new_vio = MIN_SCORE;
            return;
        }

        // Skip processing if either vector is empty
        if (c.empty() || w_no_c.empty()) {
            new_vio = MIN_SCORE;
            return;
        }

        // Pre-allocate with reserve to avoid reallocation
        std::vector<int> tmp_c;
        tmp_c.reserve(c.size());
        tmp_c = c;  // Single copy

        double best_vio = MIN_SCORE;
        std::pair<int, int> best_swap{-1, -1};

        // Process original elements in chunks for better cache utilization
        constexpr int CHUNK_SIZE = 16;  // Adjust based on cache line size
        const int c_size = static_cast<int>(c.size());

        for (int chunk_start = 0; chunk_start < c_size;
             chunk_start += CHUNK_SIZE) {
            const int chunk_end = std::min(chunk_start + CHUNK_SIZE, c_size);

            for (int i = chunk_start; i < chunk_end; ++i) {
                const int original = c[i];
                double pos_best_vio = MIN_SCORE;
                int best_candidate = -1;

                // Process candidates
                for (const int candidate : w_no_c) {
                    if (candidate == original) {
                        continue;
                    }

                    tmp_c[i] = candidate;
                    double vio;
                    exactFindBestPermutationForOnePlan(tmp_c, plan_idx, vio);

                    if (vio > pos_best_vio) {
                        pos_best_vio = vio;
                        best_candidate = candidate;
                    }
                }

                // Restore original value
                tmp_c[i] = original;

                // Update global best if better violation found
                if (pos_best_vio > best_vio) {
                    best_vio = pos_best_vio;
                    best_swap = {original, best_candidate};
                }
            }
        }

        new_vio = best_vio;
        swap_i_j = best_swap;
    }

    ////////////////////////////////////////

    void operationsCrazy(Rank1MultiLabel &label, int i) {
        constexpr double MIN_SCORE = -std::numeric_limits<double>::max();
        std::array<MoveResult, 5> moves{
            MoveResult{label.vio},  // No operation
            MoveResult{},           // Add
            MoveResult{},           // Remove
            MoveResult{},           // Swap
        };

        // Thread-local storage for temporary data
        thread_local std::vector<int> tmp_c;
        thread_local std::vector<int> new_c;
        thread_local std::vector<int> new_w_no_c;

        double new_vio = MIN_SCORE;
        const bool can_add = label.search_dir == 'a' || label.search_dir == 's';
        const bool can_remove =
            label.search_dir == 'r' || label.search_dir == 's';
        const bool can_swap =
            label.search_dir == 'a' || label.search_dir == 'r';

        // Use mutex for operations that need to modify shared data
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            // Handle operations based on search direction
            // Perform valid operations
            if (can_add) {
                int add_j;
                addSearchCrazy(label.plan_idx, label.c, label.w_no_c, new_vio,
                               add_j);
                moves[1] = MoveResult{new_vio};
                moves[1].operation_data = add_j;
            }

            if (can_remove) {
                int remove_j;
                removeSearchCrazy(label.plan_idx, label.c, new_vio, remove_j);
                moves[2] = MoveResult{new_vio};
                moves[2].operation_data = remove_j;
            }

            if (can_swap) {
                std::pair<int, int> swap_pair;
                swapSearchCrazy(label.plan_idx, label.c, label.w_no_c, new_vio,
                                swap_pair);
                moves[3] = MoveResult{new_vio};
                moves[3].operation_data = swap_pair;
            }
        }

        // Find best move
        const auto best_move_it = std::max_element(
            moves.begin(), moves.end(), [](const auto &a, const auto &b) {
                return a.violation_score < b.violation_score;
            });
        const size_t best_idx = std::distance(moves.begin(), best_move_it);
        const auto &best_move = *best_move_it;

        // Handle no operation case with thread safety
        if (best_idx == 0) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            cutLong tmp;
            for (const int j : label.c) {
                tmp.set(j);
            }
            generated_rank1_multi_pool[label.c.size()].emplace_back(
                tmp, label.plan_idx, best_move.violation_score);
            return;
        }

        // Execute the chosen operation using std::visit with thread safety
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            std::visit(
                overloaded{[](const std::monostate &) { /* No operation */ },
                           [&](const int j) {
                               if (best_idx == 1) {  // Add
                                   label.w_no_c.erase(
                                       std::remove(label.w_no_c.begin(),
                                                   label.w_no_c.end(), j),
                                       label.w_no_c.end());
                                   label.c.push_back(j);
                               } else {  // Remove
                                   label.c.erase(std::remove(label.c.begin(),
                                                             label.c.end(), j),
                                                 label.c.end());
                               }
                           },
                           [&](const std::pair<int, int> &swap_pair) {
                               const auto [from, to] = swap_pair;
                               if (auto it = std::find(label.c.begin(),
                                                       label.c.end(), from);
                                   it != label.c.end()) {
                                   *it = to;
                                   label.w_no_c.erase(
                                       std::remove(label.w_no_c.begin(),
                                                   label.w_no_c.end(), to),
                                       label.w_no_c.end());
                               }
                           }},
                best_move.operation_data);

            label.vio = best_move.violation_score;
        }
    }

    // Helper struct for std::visit
    template <class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    void chooseCuts(const std::vector<R1c> &tmp_cuts,
                    std::vector<R1c> &chosen_cuts, int numCuts) {
        // Early exit and capacity optimization
        numCuts = std::min(numCuts, static_cast<int>(tmp_cuts.size()));
        if (numCuts == 0) return;
        chosen_cuts.reserve(numCuts);

        // Preallocate vectors to avoid repeated allocations
        std::vector<std::vector<int>> tmp_cut;
        tmp_cut.reserve(
            32);  // Reasonable initial capacity for coefficient groups
        std::vector<int> new_cut;
        new_cut.reserve(1024);  // Reasonable initial capacity for combined cuts

        for (const auto &cut : tmp_cuts) {
            const auto &fst = cut.info_r1c.first;
            const auto &snd = cut.info_r1c.second;
            const int size = static_cast<int>(fst.size());
            const auto &coeff = get<0>(map_rank1_multiplier[size][snd]);

            // Find max coefficient to minimize resizing
            const int max_coeff = coeff[0];

            // Clear and resize tmp_cut in one operation
            tmp_cut.clear();
            tmp_cut.resize(max_coeff + 1);

            // Single-pass coefficient grouping
            for (int i = 0; i < size; ++i) {
                tmp_cut[coeff[i]].push_back(fst[i]);
            }

            // Sort groups and combine in one pass
            new_cut.clear();
            for (int i = max_coeff; i >= 0; --i) {
                auto &group = tmp_cut[i];
                if (group.size() > 1) {
                    pdqsort(group.begin(), group.end());
                }
                new_cut.insert(new_cut.end(),
                               std::make_move_iterator(group.begin()),
                               std::make_move_iterator(group.end()));
            }

            // Move the new cut into chosen_cuts
            chosen_cuts.emplace_back(
                R1c{std::make_pair(std::move(new_cut), snd)});

            if (--numCuts == 0) break;
        }
    }

    void constructCutsCrazy() {
        // Preallocate sets with reasonable initial sizes
        ankerl::unordered_dense::set<cutLong> cut_set;
        cut_set.reserve(max_num_r1c_per_round * 2);

        ankerl::unordered_dense::set<int> p_set;
        p_set.reserve(max_num_r1c_per_round);

        // Preallocate vector for temporary cuts
        std::vector<R1c> tmp_cuts;
        tmp_cuts.reserve(max_num_r1c_per_round);

        // Iterate over each pool of generated cuts
        for (auto &pool : generated_rank1_multi_pool) {
            auto &cuts_in_pool = pool.second;
            if (cuts_in_pool.empty()) continue;

            // Sort cuts by violation score in descending order
            pdqsort(cuts_in_pool.begin(), cuts_in_pool.end(),
                    [](const auto &a, const auto &b) {
                        return std::get<2>(a) >
                               std::get<2>(b);  // Sort by violation score
                    });

            // Calculate the violation threshold
            const double vio_threshold =
                std::get<2>(cuts_in_pool.front()) * cut_vio_factor;

            // Clear the temporary cuts vector but retain its capacity
            tmp_cuts.clear();
            int num_cuts = 0;

            // Process cuts up to the violation threshold or max count
            for (const auto &cut : cuts_in_pool) {
                // Stop if the violation score is below the threshold or max
                // cuts are reached
                if (std::get<2>(cut) < vio_threshold ||
                    num_cuts >= max_num_r1c_per_round) {
                    break;
                }

                const auto &key = std::get<0>(cut);     // Cut key (Bitset)
                const int plan_idx = std::get<1>(cut);  // Plan index

                // Skip if the cut or plan has already been processed
                if (!cut_set.insert(key).second ||
                    !p_set.insert(plan_idx).second) {
                    continue;
                }

                // Look up the cut plans
                std::vector<std::pair<std::vector<int>, double>> cut_plans;
                if (map_cut_plan_vio.find(key, cut_plans)) {
                    // Ensure the plan index is valid
                    if (plan_idx >= 0 &&
                        static_cast<size_t>(plan_idx) < cut_plans.size()) {
                        const auto &cut_plan = cut_plans[plan_idx];
                        // Validate the cut plan
                        if (!cut_plan.first.empty()) {
                            // Add the cut to the temporary cuts vector
                            tmp_cuts.emplace_back(
                                R1c{std::make_pair(cut_plan.first, plan_idx)});
                            ++num_cuts;
                        }
                    }
                }
            }

            // Process the accumulated cuts
            if (!tmp_cuts.empty()) {
                chooseCuts(tmp_cuts, cuts, max_num_r1c_per_round);
            }
        }
    }

    void findPlanForRank1Multi(
        const std::vector<int> &vis, const int denominator, cutLong &mem,
        std::vector<ankerl::unordered_dense::set<int>> &segment,
        std::vector<std::vector<int>> &plan) {
        // Preallocate space for key
        std::vector<int> key;
        key.reserve(vis.size() + 1);

        // Calculate initial sum and build key in one pass
        int sum = 0;
        key = vis;  // Copy vis first
        for (int val : vis) {
            sum += val;
        }
        key.push_back(sum % denominator);

        // Get cached result reference
        auto &cached_result = rank1_multi_mem_plan_map[key];

        // Generate solutions if not cached
        if (cached_result.empty()) {
            std::deque<State> states;
            states.emplace_back(0, sum % denominator, vis, std::vector<int>{});

            // Preallocate vectors for state processing
            std::vector<int> new_remaining;
            new_remaining.reserve(vis.size());

            while (!states.empty()) {
                auto [begin, target_rem, remaining, memory] =
                    std::move(states.front());
                states.pop_front();

                int running_sum = 0;
                const size_t remaining_size = remaining.size();

                for (size_t j = 0; j < remaining_size; ++j) {
                    running_sum = (running_sum + remaining[j]) % denominator;

                    if (running_sum > 0) {
                        const size_t current_pos = begin + j;

                        // Branch only if conditions are met
                        if (running_sum <= target_rem &&
                            j + 1 < remaining_size) {
                            new_remaining.clear();
                            new_remaining.insert(new_remaining.end(),
                                                 remaining.begin() + j + 1,
                                                 remaining.end());

                            states.emplace_back(
                                current_pos + 1, target_rem - running_sum,
                                std::move(new_remaining), memory);
                        }

                        // Add to memory if not last element
                        if (current_pos != vis.size() - 1) {
                            memory.push_back(current_pos);
                        }
                    }
                }
                cached_result.push_back(std::move(memory));
            }
        }

        // Process memory constraints
        for (int i = 1; i < dim; ++i) {
            if (mem[i]) {
                for (auto &seg : segment) {
                    seg.erase(i);
                }
            }
        }

        // Build visibility map with preallocation
        std::vector<ankerl::unordered_dense::set<int>> vertex_visibility(dim);
        for (auto &set : vertex_visibility) {
            set.reserve(cached_result.size());
        }

        for (size_t plan_idx = 0; plan_idx < cached_result.size(); ++plan_idx) {
            for (int j : cached_result[plan_idx]) {
                for (int k : segment[j]) {
                    vertex_visibility[k].insert(plan_idx);
                }
            }
        }

        // Update memory based on visibility
        const size_t cached_size = cached_result.size();
        for (int i = 1; i < dim; ++i) {
            if (vertex_visibility[i].size() == cached_size) {
                mem.set(i);
                for (auto &seg : segment) {
                    seg.erase(i);
                }
            }
        }

        // Process memory patterns with preallocation
        std::vector<std::pair<cutLong, std::vector<int>>> memory_patterns;
        memory_patterns.reserve(cached_result.size());

        for (const auto &memory_indices : cached_result) {
            cutLong pattern_mem;

            // Build memory pattern
            for (int j : memory_indices) {
                for (int k : segment[j]) {
                    pattern_mem.set(k);
                }
            }

            // Early exit for empty pattern
            if (pattern_mem.none()) {
                plan.clear();
                return;
            }

            // Find matching pattern using linear search since Bitset lacks <
            // operator
            auto it = std::find_if(
                memory_patterns.begin(), memory_patterns.end(),
                [&pattern_mem](const auto &entry) {
                    return ((pattern_mem & entry.first) ^ pattern_mem).none();
                });

            if (it != memory_patterns.end()) {
                it->first = std::move(pattern_mem);
                it->second = memory_indices;
            } else {
                memory_patterns.emplace_back(std::move(pattern_mem),
                                             memory_indices);
            }
        }

        // Build final plan with move semantics
        plan.resize(memory_patterns.size());
        std::transform(std::make_move_iterator(memory_patterns.begin()),
                       std::make_move_iterator(memory_patterns.end()),
                       plan.begin(),
                       [](auto &&entry) { return std::move(entry.second); });
    }

    void constructMemoryVertexBased() {
        // Pre-allocate memory set with dimension size
        ankerl::unordered_dense::set<int> mem;
        mem.reserve(dim);

        // Process cuts sequentially
        for (auto &c : cuts) {
            auto &cut = c.info_r1c;

            // Skip if cut.second is zero
            if (cut.second == 0) continue;

            // Create a new set for each iteration to avoid clearing overhead
            ankerl::unordered_dense::set<int> local_mem;
            local_mem.reserve(dim);

            bool if_suc = false;

            findMemoryForRank1Multi(cut, local_mem, if_suc);

            if (if_suc) {
                c.arc_mem.assign(local_mem.begin(), local_mem.end());
            }
        }
    }

    void combinations(
        const std::vector<std::vector<std::vector<int>>> &array,
        const std::vector<std::vector<ankerl::unordered_dense::set<int>>>
            &vec_segment,
        int i, std::vector<int> &accum,
        const ankerl::unordered_dense::set<int> &mem, int &record_min,
        ankerl::unordered_dense::set<int> &new_mem) {
        // Base case: all levels have been processed
        if (i == array.size()) {
            int num = 0;
            ankerl::unordered_dense::set<int> tmp_mem =
                mem;  // Copy `mem` for this combination
            tmp_mem.reserve(mem.size() +
                            10);  // Reserve memory to reduce reallocations

            // Evaluate the current combination
            for (int j = 0; j < array.size(); ++j) {
                for (int k : array[j][accum[j]]) {
                    for (int l : vec_segment[j][k]) {
                        if (tmp_mem.insert(l).second) {  // Only count if a new
                                                         // element was inserted
                            ++num;
                            if (num >= record_min) {
                                return;  // Early exit if this path already
                                         // exceeds the minimum
                            }
                        }
                    }
                }
            }

            // Update `record_min` and `new_mem` if a new minimum was found
            if (num < record_min) {
                record_min = num;
                new_mem =
                    std::move(tmp_mem);  // Use move semantics for efficiency
            }
        } else {
            // Preallocate `accum` to avoid repeated `push_back` and `pop_back`
            if (accum.size() <= i) {
                accum.resize(i + 1);
            }

            // Iterate through choices for the current level `i`
            for (int j = 0; j < array[i].size(); ++j) {
                accum[i] = j;  // Update the current level's choice
                combinations(array, vec_segment, i + 1, accum, mem, record_min,
                             new_mem);
            }
        }
    }

    void findMemoryForRank1Multi(
        const std::pair<std::vector<int>, int> &cut_pair,
        ankerl::unordered_dense::set<int> &mem, bool &if_suc) {
        if_suc = true;
        const auto &cut = cut_pair.first;
        int plan_idx = cut_pair.second;
        int size = static_cast<int>(cut.size());

        const auto &multi = std::get<0>(map_rank1_multiplier[size][plan_idx]);
        int denominator = std::get<1>(map_rank1_multiplier[size][plan_idx]);

        // Calculate visit times based on `cut` and `multi`
        std::vector<int> num_vis_times(sol.size(), 0);
        ankerl::unordered_dense::map<int, int> map_cut_mul;
        for (int i = 0; i < cut.size(); ++i) {
            int vertex = cut[i];
            int multiplier = multi[i];
            map_cut_mul[vertex] = multiplier;

            for (int route_idx = 0; route_idx < v_r_map[vertex].size();
                 ++route_idx) {
                int frequency = v_r_map[vertex][route_idx];
                if (frequency >
                    0) {  // Only proceed if the vertex appears in this route
                    num_vis_times[route_idx] += multiplier * frequency;
                }
            }
        }

        // Divide `num_vis_times` by `denominator`
        std::transform(num_vis_times.begin(), num_vis_times.end(),
                       num_vis_times.begin(),
                       [denominator](int x) { return x / denominator; });

        cutLong mem_long;
        std::vector<std::vector<std::vector<int>>> vec_data;
        std::vector<std::vector<ankerl::unordered_dense::set<int>>>
            vec_segment_route;

        // Reserve memory for `vec_data` and `vec_segment_route`
        vec_data.reserve(sol.size());
        vec_segment_route.reserve(sol.size());

        // Populate `vec_data` and `vec_segment_route`
        for (int num = 0; const auto &route : sol) {
            if (num_vis_times[num++] == 0) continue;

            std::vector<int> vis;
            std::vector<ankerl::unordered_dense::set<int>> segment_route;
            ankerl::unordered_dense::set<int> tmp_seg;

            for (int v : route.route) {
                if (auto it = map_cut_mul.find(v); it != map_cut_mul.end()) {
                    vis.push_back(it->second);
                    segment_route.push_back(std::move(tmp_seg));
                    tmp_seg.clear();
                } else {
                    tmp_seg.insert(v);
                }
            }
            if (!segment_route.empty()) {
                segment_route.erase(
                    segment_route.begin());  // Remove first segment
            }

            std::vector<std::vector<int>> data;
            findPlanForRank1Multi(vis, denominator, mem_long, segment_route,
                                  data);
            if (!data.empty()) {
                vec_data.push_back(std::move(data));
                vec_segment_route.push_back(std::move(segment_route));
            }
        }

        // Filter `vec_data` and `vec_segment_route` based on `mem_long`
        auto filter = [&mem_long](const auto &data, const auto &segments) {
            return std::any_of(
                data.begin(), data.end(), [&](const auto &segment) {
                    return std::all_of(
                        segment.begin(), segment.end(), [&](int idx) {
                            return std::all_of(
                                segments[idx].begin(), segments[idx].end(),
                                [&](int v) { return mem_long[v]; });
                        });
                });
        };

        auto it_data = vec_data.begin();
        auto it_segments = vec_segment_route.begin();
        while (it_data != vec_data.end()) {
            if (filter(*it_data, *it_segments)) {
                it_data = vec_data.erase(it_data);
                it_segments = vec_segment_route.erase(it_segments);
            } else {
                ++it_data;
                ++it_segments;
            }
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
        size_t cnt = std::accumulate(
            vec_data.begin(), vec_data.end(), 1,
            [](size_t acc, const auto &data) { return acc * data.size(); });
        if (cnt > 1) {
            std::vector<int> tmp;
            ankerl::unordered_dense::set<int> new_mem;
            int record_min = std::numeric_limits<int>::max();
            combinations(vec_data, vec_segment_route, 0, tmp, mem, record_min,
                         new_mem);
            mem = std::move(new_mem);
        }
    }

    void printPool() {
        for (const auto &i : rank1_multi_label_pool) {
            fmt::print("Cut: ");
            for (int node : i.c) {
                fmt::print("{} ", node);
            }
            fmt::print("| Plan Index: {}\n", i.plan_idx);
        }
    }

    void findMemAggressively(
        const std::vector<std::vector<std::vector<int>>> &array,
        const std::vector<std::vector<ankerl::unordered_dense::set<int>>>
            &vec_segment,
        ankerl::unordered_dense::set<int> &mem) {
        mem.reserve(array.size() * 10);  // Adjust reserve size if more
                                         // precise estimates are possible

        for (int i = 0; i < array.size(); ++i) {
            const auto &r = array[i];
            const auto &segments = vec_segment[i];

            // Cache sizes to avoid recalculating in the inner loop
            std::vector<int> segment_sizes(r.size());
            for (int j = 0; j < r.size(); ++j) {
                segment_sizes[j] = std::accumulate(
                    r[j].begin(), r[j].end(), 0, [&](int sum, int k) {
                        return sum + static_cast<int>(segments[k].size());
                    });
            }

            // Find the index of the minimum size segment
            int min_idx = std::distance(
                segment_sizes.begin(), std::ranges::min_element(segment_sizes));

            // Manually insert elements from the minimum index segment into
            // `mem`
            for (int k : r[min_idx]) {
                mem.insert(segments[k].begin(), segments[k].end());
            }
        }
    }

   private:
};
