/**
 * @file VRPNode.h
 * @brief This file contains the definition of the VRPNode struct.
 *
 * This file contains the definition of the VRPNode struct, which represents a
 * node in a Vehicle Routing Problem. The struct contains information about a
 * node, such as its ID, start time, end time, duration, cost, demand, and
 * capacity constraints. It provides constructors to initialize the node with
 * different sets of parameters. The `setDuals` method allows updating the cost
 * of the node.
 *
 */
#pragma once
#include "Arc.h"
#include "Common.h"
#include "Cut.h"
#include "Definitions.h"
/**
 * @struct VRPNode
 * @brief Represents a node in a Vehicle Routing Problem.
 *
 * This struct contains information about a node, such as its ID, start time,
 * end time, duration, cost, demand, and capacity constraints. It provides
 * constructors to initialize the node with different sets of parameters. The
 * `setDuals` method allows updating the cost of the node.
 */
struct VRPNode {
    double x;
    double y;
    int id;
    double start_time;
    double end_time;
    double duration;
    double cost = 0.0;
    double demand;
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<double> mtw_lb;
    std::vector<double> mtw_ub;
    std::vector<Arc> fw_arcs;
    std::vector<Arc> bw_arcs;
    std::vector<std::vector<Arc>> fw_arcs_scc;
    std::vector<std::vector<Arc>> bw_arcs_scc;
    std::vector<JumpArc> fw_jump_arcs;
    std::vector<JumpArc> bw_jump_arcs;
    std::string identifier;

    bool is_station = false;

    std::string track_id;
    int subject;
    int week;
    int year;
    double duration_min;
    int setup_time;
    int teardown_time;
    long long time_window_start;
    long long time_window_end;

    std::vector<double> consumption;

    // default constructor
    VRPNode() = default;

    VRPNode(int i, int st, int et, int dur, double c)
        : id(i), start_time(st), end_time(et), duration(dur), cost(c) {}

    VRPNode(int i, int st, int et, int dur, double c, double d)
        : id(i),
          start_time(st),
          end_time(et),
          duration(dur),
          cost(c),
          demand(d) {}

    // -----------------------------
    // SCC Computation Support
    // -----------------------------
    /**
     * @brief Compute the strongly connected components (SCCs) for a VRP graph.
     *
     * Given a vector of VRPNodes, this function builds a graph using each
     * node's forward arcs (assumed to represent directed edges from node.id to
     * arc.to), and runs Tarjan's algorithm to compute the SCCs.
     *
     * @param nodes The vector of VRPNodes representing the VRP.
     * @return A vector of SCCs, each SCC is a vector of node IDs.
     */
    static std::vector<std::vector<int>> computeSCCs(
        const std::vector<VRPNode> &nodes) {
        int V = nodes.size();
        // Build adjacency list: vertices are nodes (using node.id) and an edge
        // exists from u to v if there is a forward arc from node u to node v.
        std::vector<std::vector<int>> graph(V);
        for (const auto &node : nodes) {
            for (const auto &arc : node.fw_arcs) {
                // Assumes that node.id is in [0, V-1] and arc.to corresponds to
                // another node's id.
                graph[node.id].push_back(arc.to);
            }
        }

        std::vector<int> disc(V, -1), low(V, -1);
        std::vector<bool> inStack(V, false);
        std::stack<int> st;
        std::vector<std::vector<int>> sccs;
        int time = 0;

        // Tarjan's recursive helper.
        std::function<void(int)> tarjanUtil = [&](int u) {
            disc[u] = low[u] = time++;
            st.push(u);
            inStack[u] = true;

            for (int v : graph[u]) {
                if (disc[v] == -1) {
                    tarjanUtil(v);
                    low[u] = std::min(low[u], low[v]);
                } else if (inStack[v]) {
                    low[u] = std::min(low[u], disc[v]);
                }
            }
            if (low[u] == disc[u]) {
                std::vector<int> scc;
                while (true) {
                    int w = st.top();
                    st.pop();
                    inStack[w] = false;
                    scc.push_back(w);
                    if (w == u) break;
                }
                sccs.push_back(scc);
            }
        };

        for (int i = 0; i < V; i++) {
            if (disc[i] == -1) tarjanUtil(i);
        }
        return sccs;
    }
    int sccId = -1;

    /**
     * @brief Assigns SCC IDs to a vector of VRPNodes.
     *
     * This method computes the SCCs for the given VRP nodes and then assigns
     * each node an SCC identifier (starting from 0) based on the SCC it belongs
     * to.
     *
     * @param nodes The vector of VRPNodes to update.
     */
    static void assignSCCIds(std::vector<VRPNode> &nodes) {
        auto sccs = computeSCCs(nodes);
        for (size_t id = 0; id < sccs.size(); id++) {
            for (int nodeId : sccs[id]) {
                nodes[nodeId].sccId = id;
            }
        }
    }

    template <Direction D>
    void sort_arcs_by_scores(
        const ankerl::unordered_dense::map<Arc, int, arc_hash> &arc_scores,
        std::vector<VRPNode> &nodes, std::vector<ActiveCutInfo> &cuts) {
        // Helper lambda: compute a score for a given arc.
        // The base score is defined as the negative cost (scaled) of the
        // destination node. If the arc is found in arc_scores, add its
        // bonus/penalty. If any cut applies (i.e. the arc is in the SRC set for
        // the cut), override the score.
        auto get_score = [&arc_scores, &nodes,
                          &cuts](const Arc &arc) -> double {
            // Start with a base score as before.
            double score = nodes[arc.to].cost / 10.0;
            if (auto it = arc_scores.find(arc); it != arc_scores.end()) {
                score += it->second;
            }
            for (const auto &cut : cuts) {
                if (cut.type == CutType::ThreeRow &&
                    cut.isSRCset(arc.from, arc.to)) {
                    score += cut.dual_value / 10.0;
                    break;
                }
            }
            // // Incorporate SCC information.
            // // For example, if the source and destination are in different
            // SCCs,
            // // subtract a penalty.
            // if (nodes[arc.from].sccId != nodes[arc.to].sccId) {
            //     // The penalty value can be tuned.
            //     score -= 5.0;
            // } else {
            //     // Optionally, add a bonus if within the same SCC.
            //     score += 1.0;
            // }
            return score;
        };

        // --- Sort arcs within strongly connected components ---
        // (Assumes that if D==Direction::Forward, fw_arcs_scc holds the forward
        // SCCs; otherwise, bw_arcs_scc holds the backward SCCs.)
        // if constexpr (D == Direction::Forward) {
        //     for (auto &arcs : fw_arcs_scc) {
        //         pdqsort(arcs.begin(), arcs.end(),
        //                 [&](const Arc &a, const Arc &b) {
        //                     return get_score(a) > get_score(b);
        //                 });
        //     }
        // } else {
        //     for (auto &arcs : bw_arcs_scc) {
        //         pdqsort(arcs.begin(), arcs.end(),
        //                 [&](const Arc &a, const Arc &b) {
        //                     return get_score(a) > get_score(b);
        //                 });
        //     }
        // }

        // // --- Sort the main list of arcs ---
        // if constexpr (D == Direction::Forward) {
        //     pdqsort(fw_arcs.begin(), fw_arcs.end(),
        //             [&](const Arc &a, const Arc &b) {
        //                 return get_score(a) > get_score(b);
        //             });
        // } else {
        //     pdqsort(bw_arcs.begin(), bw_arcs.end(),
        //             [&](const Arc &a, const Arc &b) {
        //                 return get_score(a) > get_score(b);
        //             });
        // }
    }

    /**
     * @brief Adds an arc between two buckets with specified resource increments
     * and cost.
     *
     * This function adds a forward or backward arc between the specified
     * buckets. The arc is characterized by resource increments and a cost
     * increment.
     *
     */
    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc,
                 double cost_inc, bool fw) {
        if (fw) {
            fw_arcs.push_back(
                {from_bucket, to_bucket, std::move(res_inc), cost_inc});
        } else {
            bw_arcs.push_back(
                {from_bucket, to_bucket, std::move(res_inc), cost_inc});
        }
    }

    /**
     * @brief Adds an arc to the forward or backward arc list.
     *
     * This function adds an arc between two buckets, either to the forward arc
     * list or the backward arc list, based on the direction specified by the
     * `fw` parameter.
     *
     */
    template <Direction D>

    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc,
                 double cost_inc, bool fixed) {
        if constexpr (D == Direction::Forward) {
            fw_arcs.push_back(
                {from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        } else {
            bw_arcs.push_back(
                {from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        }
    }

    template <Direction D>
    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc,
                 double cost_inc, double priority) {
        if constexpr (D == Direction::Forward) {
            fw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc),
                               cost_inc, priority});
        } else {
            bw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc),
                               cost_inc, priority});
        }
    }

    template <Direction D>
    void add_jump_arc(int from_bucket, int to_bucket,
                      const std::vector<double> &res_inc, double cost_inc,
                      int to_job = -1) {
        if constexpr (D == Direction::Forward) {
            fw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc,
                                      to_job);
        } else {
            bw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc,
                                      to_job);
        }
    }

    /**
     * @brief Sorts the forward and backward arcs based on their priority.
     *
     * This function sorts the `fw_arcs` in descending order of priority and
     * the `bw_arcs` in ascending order of priority.
     */
    void sort_arcs() {
        std::sort(
            fw_arcs.begin(), fw_arcs.end(),
            [](const Arc &a, const Arc &b) { return a.priority > b.priority; });

        std::sort(
            bw_arcs.begin(), bw_arcs.end(),
            [](const Arc &a, const Arc &b) { return a.priority < b.priority; });
    }

    /**
     * @brief Sets the location coordinates.
     *
     * This function sets the x and y coordinates for the location.
     */
    void set_location(double x, double y) {
        this->x = x;
        this->y = y;
    }

    /**
     * @brief Retrieves a constant reference to the vector of arcs based on the
     * specified direction.
     *
     * This function template returns a constant reference to either the forward
     * arcs or backward arcs vector, depending on the direction specified by the
     * template parameter.
     *
     */
    template <Direction dir>
    inline std::span<const Arc> get_arcs() const {
        return (dir == Direction::Forward) ? fw_arcs : bw_arcs;
    }

    template <Direction dir>
    inline auto get_jump_arcs(int to_job) const {
        const auto &jump_arcs =
            (dir == Direction::Forward) ? fw_jump_arcs : bw_jump_arcs;
        // Directly construct the filter view
        return std::ranges::views::filter(
            jump_arcs,
            [to_job](const JumpArc &arc) { return arc.to_job == to_job; });
    }

    /**
     * @brief Retrieves the arcs associated with a given strongly connected
     * component (SCC) in the specified direction.
     *
     */
    template <Direction dir>
    inline std::span<const Arc> get_arcs(int scc) const {
        const auto &arcs =
            (dir == Direction::Forward) ? fw_arcs_scc[scc] : bw_arcs_scc[scc];
        return std::span<const Arc>(arcs);
    }

    /**
     * @brief Clears all forward and backward arcs.
     *
     * This function empties the containers holding the forward arcs (fw_arcs)
     * and backward arcs (bw_arcs), effectively removing all stored arcs.
     */
    void clear_arcs() {
        fw_arcs.clear();
        bw_arcs.clear();
    }

    template <Direction dir>
    void clear_jump_arcs() {
        if constexpr (dir == Direction::Forward) {
            fw_jump_arcs.clear();
        } else {
            bw_jump_arcs.clear();
        }
    }

    // define setDuals method
    void setDuals(double d) { cost = d; }
};
