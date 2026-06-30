/**
 * @file BucketTopology.h
 * @brief Bucket arc generation and SCC/topological ordering.
 */

#pragma once

#include "BucketGraph.h"
#include "utils/NumericUtils.h"

template <Direction D>
void BucketGraph::generate_arcs() {
    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear();
        fw_arcs.clear();
    } else {
        bw_bucket_graph.clear();
        bw_arcs.clear();
    }

    auto &buckets         = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets     = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_idx = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

    const int                        num_resources = options.resources.size();
    std::vector<std::vector<double>> node_base_intervals(nodes.size());
    std::vector<std::vector<int>>    node_split_counts(nodes.size());

    for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
        const auto &node = nodes[node_id];
        node_base_intervals[node_id].resize(num_resources);
        node_split_counts[node_id].resize(num_resources);
        for (int r = 0; r < num_resources; ++r) {
            double full_range = R_max[r] - R_min[r];
            double node_range = node.ub[r] - node.lb[r];
            int    splits     = 1;
            if (std::fabs(full_range) > std::numeric_limits<double>::epsilon()) {
                double splits_d = (node_range * intervals[r].interval) / full_range;
                splits          = std::max(1, static_cast<int>(std::round(splits_d)));
            }
            node_split_counts[node_id][r]   = splits;
            node_base_intervals[node_id][r] = (node.ub[r] - node.lb[r]) / static_cast<double>(splits);
        }
    }

    for (auto &bucket : buckets) {
        bucket.clear();
        bucket.clear_arcs(D == Direction::Forward);
    }

    auto try_add_arc = [&](int from_bucket, const VRPNode &next_node, const std::vector<double> &res_inc,
                           double cost_inc) -> bool {
        bool                valid = true;
        std::vector<double> head_resource(res_inc.size(), 0.0);
        if constexpr (D == Direction::Forward) {
            for (int r = 0; r < res_inc.size() && valid; ++r) {
                if (numericutils::gt(buckets[from_bucket].lb[r] + res_inc[r], next_node.ub[r])) {
                    valid = false;
                } else {
                    head_resource[r] = std::max(buckets[from_bucket].lb[r] + res_inc[r], next_node.lb[r]);
                }
            }
        } else {
            for (int r = 0; r < res_inc.size() && valid; ++r) {
                if (numericutils::lt(buckets[from_bucket].ub[r] - res_inc[r], next_node.lb[r])) {
                    valid = false;
                } else {
                    head_resource[r] = std::min(buckets[from_bucket].ub[r] - res_inc[r], next_node.ub[r]);
                }
            }
        }
        if (valid) {
            const int to_bucket = get_bucket_number<D>(next_node.id, head_resource);
            buckets[from_bucket].template add_bucket_arc<D>(from_bucket, to_bucket, res_inc, cost_inc, false);
        }
        return valid;
    };

    auto process_node = [&](int node_id) {
        const auto         &node = nodes[node_id];
        std::vector<double> res_inc(num_resources);
        const auto          arcs = node.get_arcs<D>();

        for (int i = 0; i < num_buckets[node.id]; ++i) {
            int from_bucket = i + num_buckets_idx[node.id];
            for (const auto &arc : arcs) {
                const auto &next_node = nodes[arc.to];
                if (node.id == next_node.id) continue;

                const double travel_cost = getcij(node.id, next_node.id);
                double       cost_inc    = travel_cost - next_node.cost;
                for (int r = 0; r < num_resources; r++) {
                    res_inc[r] = node.consumption[r];
                    if (options.resources[r] == "time") { res_inc[r] += travel_cost; }
                }

                try_add_arc(from_bucket, next_node, res_inc, cost_inc);
            }
        }
    };

    std::vector<int> tasks(nodes.size());
    std::iota(tasks.begin(), tasks.end(), 0);

    constexpr int          chunk_size   = 10;
    const unsigned int     thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    exec::static_thread_pool pool(thread_count);
    auto                     sched = pool.get_scheduler();

    auto bulk_sender = stdexec::bulk(stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
                                     [&, chunk_size](std::size_t chunk_idx) {
                                         const size_t start_idx = chunk_idx * chunk_size;
                                         const size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());
                                         for (size_t idx = start_idx; idx < end_idx; ++idx) {
                                             process_node(tasks[idx]);
                                         }
                                     });

    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);
    auto &global_arcs  = assign_buckets<D>(fw_arcs, bw_arcs);
    bucket_graph.assign(buckets.size(), {});
    global_arcs.clear();

    size_t arc_count = 0;
    for (int b = 0; b < static_cast<int>(buckets.size()); ++b) {
        const auto &bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        arc_count += bucket_arcs.size();
    }
    global_arcs.reserve(arc_count);

    for (int b = 0; b < static_cast<int>(buckets.size()); ++b) {
        auto       &neighbors   = bucket_graph[b];
        const auto &bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        neighbors.reserve(bucket_arcs.size());
        for (const auto &arc : bucket_arcs) {
            neighbors.push_back(arc.to_bucket);
            global_arcs.push_back(arc);
        }
    }
}

template <Direction D>
Label *BucketGraph::get_best_label(const std::vector<int> &, const std::vector<double> &,
                                   const std::vector<std::vector<int>> &) {
    double best_cost  = std::numeric_limits<double>::infinity();
    Label *best_label = nullptr;
    auto  &buckets    = assign_buckets<D>(fw_buckets, bw_buckets);

    for (auto &bucket : buckets) {
        Label *label = bucket.get_best_label();
        if (!label) continue;
        if (label->cost < best_cost) {
            best_cost  = label->cost;
            best_label = label;
        }
    }
    return best_label;
}

template <Direction D>
void BucketGraph::SCC_handler() {
    auto &Phi          = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets      = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);

    std::vector<std::vector<int>> extended_bucket_graph = bucket_graph;
    if (extended_bucket_graph.size() < buckets.size()) { extended_bucket_graph.resize(buckets.size()); }
    for (size_t i = 0; i < buckets.size(); ++i) {
        const auto &phi_set = Phi[i];
        if (phi_set.empty()) continue;
        for (auto phi_bucket : phi_set) {
            if (phi_bucket >= 0 && static_cast<size_t>(phi_bucket) < extended_bucket_graph.size()) {
                extended_bucket_graph[static_cast<size_t>(phi_bucket)].push_back(static_cast<int>(i));
            }
        }
    }

    SCC scc_finder;
    scc_finder.convertFromAdjacency(extended_bucket_graph);
    auto sccs              = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

#ifdef VERBOSE
    constexpr auto blue  = "\033[34m";
    constexpr auto reset = "\033[0m";
    fmt::print((D == Direction::Forward) ? "FW SCCs:\n" : "BW SCCs:\n");
    for (auto scc : topological_order) {
        fmt::print("{}({}) -> {}", blue, scc, reset);
        for (auto bucket : sccs[scc]) { fmt::print("{} ", bucket); }
        fmt::print("\n");
    }
    fmt::print("\n");
#endif

    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size());
    for (int scc : topological_order) { ordered_sccs.push_back(sccs[scc]); }

    auto sorted_sccs = sccs;
    for (auto &scc : sorted_sccs) {
        if constexpr (D == Direction::Forward) {
            pdqsort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].lb < buckets[b].lb; });
        } else {
            pdqsort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].ub > buckets[b].ub; });
        }
    }

    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            node.fw_arcs_scc.resize(sccs.size());
        } else {
            node.bw_arcs_scc.resize(sccs.size());
        }
    }

    auto process_bucket_arcs = [&](int bucket, int scc_index) {
        int      from_node_id = buckets[bucket].node_id;
        VRPNode &node         = nodes[from_node_id];
        const auto       &node_arcs = (D == Direction::Forward) ? node.fw_arcs : node.bw_arcs;
        std::vector<Arc> &filtered_arcs =
            (D == Direction::Forward) ? node.fw_arcs_scc[scc_index] : node.bw_arcs_scc[scc_index];

        const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
        for (const auto &arc : bucket_arcs) {
            int  to_node_id = buckets[arc.to_bucket].node_id;
            auto it         = std::find_if(node_arcs.begin(), node_arcs.end(),
                                           [&to_node_id](const Arc &a) { return a.to == to_node_id; });
            if (it != node_arcs.end()) { filtered_arcs.push_back(*it); }
        }
    };

    int scc_ctr = 0;
    for (const auto &scc : sccs) {
        for (int bucket : scc) { process_bucket_arcs(bucket, scc_ctr); }
        ++scc_ctr;
    }

    auto sortAndDedup = [](std::vector<Arc> &arcs) {
        pdqsort(arcs.begin(), arcs.end(),
                [](const Arc &a, const Arc &b) { return std::tie(a.from, a.to) < std::tie(b.from, b.to); });
        auto last = std::unique(arcs.begin(), arcs.end(),
                                [](const Arc &a, const Arc &b) { return a.from == b.from && a.to == b.to; });
        arcs.erase(last, arcs.end());
    };

    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            for (auto &arcs : node.fw_arcs_scc) { sortAndDedup(arcs); }
        } else {
            for (auto &arcs : node.bw_arcs_scc) { sortAndDedup(arcs); }
        }
    }

    std::vector<int> bucket_scc_rank(buckets.size(), -1);
    for (size_t rank = 0; rank < ordered_sccs.size(); ++rank) {
        for (int bucket : ordered_sccs[rank]) { bucket_scc_rank[bucket] = static_cast<int>(rank); }
    }

    UnionFind uf(ordered_sccs);
    if constexpr (D == Direction::Forward) {
        fw_ordered_sccs      = ordered_sccs;
        fw_topological_order = topological_order;
        fw_sccs              = sccs;
        fw_sccs_sorted       = sorted_sccs;
        fw_bucket_scc_rank   = std::move(bucket_scc_rank);
        fw_union_find        = uf;
    } else {
        bw_ordered_sccs      = ordered_sccs;
        bw_topological_order = topological_order;
        bw_sccs              = sccs;
        bw_sccs_sorted       = sorted_sccs;
        bw_bucket_scc_rank   = std::move(bucket_scc_rank);
        bw_union_find        = uf;
    }
}
