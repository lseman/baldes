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

    template <Direction D>
    void sort_arcs_by_scores(
        const ankerl::unordered_dense::map<Arc, int, arc_hash> &arc_scores,
        std::vector<VRPNode> &nodes) {
        auto get_score = [&arc_scores, &nodes](const Arc &arc) {
            auto it = arc_scores.find(arc);
            auto scores = -nodes[arc.to].cost / 100;
            scores += (it != arc_scores.end()) ? it->second : 0;
            return scores;
        };

        if constexpr (D == Direction::Forward) {
            for (auto &arcs : fw_arcs_scc) {
                pdqsort(arcs.begin(), arcs.end(),
                        [&](const Arc &a, const Arc &b) {
                            return get_score(a) > get_score(b);
                        });
            }
        } else {
            for (auto &arcs : bw_arcs_scc) {
                pdqsort(arcs.begin(), arcs.end(),
                        [&](const Arc &a, const Arc &b) {
                            return get_score(a) > get_score(b);
                        });
            }
        }

        if constexpr (D == Direction::Forward) {
            pdqsort(fw_arcs.begin(), fw_arcs.end(),
                    [&](const Arc &a, const Arc &b) {
                        return get_score(a) > get_score(b);
                    });
        } else {
            pdqsort(bw_arcs.begin(), bw_arcs.end(),
                    [&](const Arc &a, const Arc &b) {
                        return get_score(a) > get_score(b);
                    });
        }
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
