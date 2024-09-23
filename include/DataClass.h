#pragma once
#include <deque>
#include <iostream>
#include <tuple>
#include <vector>

#include "Definitions.h"

#pragma once
#include <deque>
#include <iostream>
#include <tuple>
#include <vector>

#include "Definitions.h"

// TODO: Add SRC and RCC penalty to the path cost
class SchrodingerPool {
private:
    std::deque<std::tuple<int, Path>> paths; // Stores tuples of (iteration added, Path)
    int                               current_iteration = 0;
    int                               max_live_time; // Max iterations a Path can stay active
    std::vector<double>               duals;         // Dual variables for each path
    std::vector<VRPJob>              *jobs = nullptr;

public:
    std::vector<std::vector<double>> distance_matrix; // Distance matrix for the graph

    SchrodingerPool(int live_time) : max_live_time(live_time) {}

    void setJobs(std::vector<VRPJob> *jobs) { this->jobs = jobs; }

    int getcij(int i, int j) { return distance_matrix[i][j]; }

    void remove_old_paths() {
        // Remove old paths that have lived beyond their allowed time
        while (!paths.empty() && std::get<0>(paths.front()) + max_live_time <= current_iteration) {
            paths.pop_front(); // Remove the oldest path
        }
    }

    void add_path(const Path &path) {
        // Add new path with the current iteration
        paths.push_back({current_iteration, path});
    }

    void add_paths(const std::vector<Path> &new_paths) {
        remove_old_paths();
        for (const Path &path : new_paths) { add_path(path); }
        computeRC();
    }

    void computeRC() {
        for (auto &path : paths) {
            int iteration_added = std::get<0>(path); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration + max_life
            if (iteration_added + max_live_time < current_iteration) { break; }

            Path &p    = std::get<1>(path);
            p.red_cost = p.cost;

            if (p.size() > 1) {
                for (int i = 0; i < p.size() - 1; i++) {
                    auto &job = (*jobs)[p[i]]; // Dereference jobs and access element
                    p.red_cost -= job.cost;
                }
            }
        }
    }

    std::vector<Path> get_paths_with_negative_red_cost() const {
        std::vector<Path> result;

        for (const auto &path_tuple : paths) {
            int iteration_added = std::get<0>(path_tuple); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration + max_life
            if (iteration_added + max_live_time < current_iteration) { break; }

            const Path &p = std::get<1>(path_tuple);

            // Add paths with negative red_cost to the result
            if (p.red_cost < 0) { result.push_back(p); }
        }

        // Sort the result based on red_cost
        std::sort(result.begin(), result.end(), [](const Path &a, const Path &b) { return a.red_cost < b.red_cost; });

        return result;
    }

    void iterate() { current_iteration++; }
};

/**
 * @struct RCCmanager
 * @brief Manages RCC cuts, dual cache, and arc-to-cut mappings with thread safety.
 *
 * This struct provides methods to add cuts, compute duals, and manage a cache of dual sums
 * for arcs. It supports parallel processing for efficiency.
 *
 */
struct RCCmanager {
    std::vector<RCCcut>                                                       cuts;
    int                                                                       cut_counter = 0;
    std::unordered_map<RCCarc, double, RCCarcHash>                            dualCache;
    std::unordered_map<std::pair<int, int>, std::vector<RCCcut *>, pair_hash> arcCutMap;
    std::mutex cache_mutex; // Protect shared resources during concurrent access

    RCCmanager() = default;

    // Add a cut to the list and update the arc-to-cut mapping
    void addCut(const std::vector<RCCarc> &arcs, double rhs, GRBConstr constr = GRBConstr()) {
        cuts.emplace_back(arcs, rhs, cut_counter++, constr);
        for (const auto &cutArc : arcs) { arcCutMap[{cutArc.from, cutArc.to}].push_back(&cuts.back()); }
    }

    // Bulk addition of cuts with parallel processing for efficiency
    void addCutBulk(const std::vector<std::vector<RCCarc>> &arcsVec, const std::vector<int> &rhsVec,
                    const std::vector<GRBConstr> &constrVec) {
        assert(arcsVec.size() == rhsVec.size() && rhsVec.size() == constrVec.size());

        cuts.reserve(cuts.size() + arcsVec.size());

        for (size_t i = 0; i < arcsVec.size(); ++i) {
            cuts.emplace_back(arcsVec[i], rhsVec[i], cut_counter++, constrVec[i]);
            for (const auto &cutArc : arcsVec[i]) {
                auto &cutList = arcCutMap[{cutArc.from, cutArc.to}];
                cutList.emplace_back(&cuts.back());
            }
        }
    }

    // Compute duals for cuts, remove small dual cuts, and update cache in parallel
    void computeDualsDeleteAndCache(GRBModel *model) {
        dualCache.clear();
        std::vector<RCCcut *> toRemoveFromCache;
        std::vector<int>      toRemoveIndices;
        const int             JOBS = 10;

        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        // Step 1: Parallel gathering of cuts for removal
        auto gather_cuts = stdexec::bulk(stdexec::just(), cuts.size(),
                                         [this, model, &toRemoveFromCache, &toRemoveIndices](std::size_t i) {
                                             RCCcut &cut = cuts[i];
                                             cut.dual    = cut.constr.get(GRB_DoubleAttr_Pi);

                                             if (std::abs(cut.dual) > 1e6 || std::isnan(cut.dual)) { cut.dual = 0.0; }

                                             if (std::abs(cut.dual) < 1e-6) {
                                                 std::lock_guard<std::mutex> lock(cache_mutex);
                                                 toRemoveFromCache.push_back(&cut);
                                                 toRemoveIndices.push_back(i);
                                             }
                                         });

        auto gather_work = stdexec::starts_on(sched, gather_cuts);
        stdexec::sync_wait(std::move(gather_work));

        // Step 2: Parallel removal of cuts from the cache
        auto remove_cuts =
            stdexec::bulk(stdexec::just(), toRemoveFromCache.size(), [this, &toRemoveFromCache](std::size_t i) {
                RCCcut *cut = toRemoveFromCache[i];
                for (const auto &cutArc : cut->arcs) {
                    auto &cutList = arcCutMap[{cutArc.from, cutArc.to}];
                    cutList.erase(std::remove(cutList.begin(), cutList.end(), cut), cutList.end());

                    if (cutList.empty()) {
                        std::lock_guard<std::mutex> lock(cache_mutex);
                        dualCache[{cutArc.from, cutArc.to}] = 0.0;
                        arcCutMap.erase({cutArc.from, cutArc.to});
                    }
                }
            });

        auto remove_work = stdexec::starts_on(sched, remove_cuts);
        stdexec::sync_wait(std::move(remove_work));

        // Step 3: Bulk erasure of cuts from the cut vector
        std::sort(toRemoveIndices.begin(), toRemoveIndices.end(), std::greater<>());
        for (int idx : toRemoveIndices) {
            // fmt::print("Erasing cut {}\n", idx);
            cuts.erase(cuts.begin() + idx);
        }

        // Step 4: Parallel dual sum computation for arcs
        auto compute_duals = stdexec::bulk(stdexec::just(), arcCutMap.size(), [this](std::size_t idx) {
            const auto &[arcKey, cutList] = *std::next(arcCutMap.begin(), idx);
            double dualSum                = 0.0;

            for (const auto *cut : cutList) {
                dualSum += cut->dual;
                if (std::abs(dualSum) > 1e6) {
                    dualSum = 0.0;
                    break;
                }
            }

            std::lock_guard<std::mutex> lock(cache_mutex);
            dualCache[{arcKey.first, arcKey.second}] = dualSum;
        });

        auto compute_duals_work = stdexec::starts_on(sched, compute_duals);
        stdexec::sync_wait(std::move(compute_duals_work));

        // Step 5: Update the model
        model->update();
    }

    // Function to retrieve cached dual sum for a given arc
    double getCachedDualSumForArc(int from, int to) {
        RCCarc arc(from, to);
        auto   it = dualCache.find(arc);
        if (it == dualCache.end()) { return 0.0; }

        double cachedDual = it->second;
        if (std::abs(cachedDual) > 1e6 || std::isnan(cachedDual)) { return 0.0; }
        return cachedDual;
    }

    // Method to retrieve all cuts (if needed)
    std::vector<RCCcut> getAllCuts() const { return cuts; }
};
