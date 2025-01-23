#pragma once
#include "BucketGraph.h"
#include "Common.h"
#include "Definitions.h"

enum ResourceType {
    Disposable = 1,
    NonDisposable = 0,
    Binary = 2,
    MTW = 3,
    Battery = 4,
    RechargeCount = 5,
    RechargeTime = 6
};

using ResourceProcessor = bool (*)(double&, double, double, const VRPNode&,
                                   const BucketOptions&);

template <Direction D>
static constexpr bool check_bounds(double val, double lb, double ub) {
    return D == Direction::Forward ? val <= ub : val >= lb;
}

template <Direction D>
static constexpr double get_binary_result(double increment) {
    return (increment > 0) ? (D == Direction::Forward ? 1.0 : 0.0)
                           : (D == Direction::Forward ? 0.0 : 1.0);
}

template <Direction D>
static constexpr double adjust_resource(double initial, double increment) {
    return initial + (D == Direction::Forward ? increment : -increment);
}

template <Direction D>
static bool process_disposable(double& new_resource, double initial,
                               double increment, const VRPNode& node,
                               const BucketOptions& opts) {
    constexpr auto dir = D == Direction::Forward;
    new_resource =
        dir ? std::max(adjust_resource<D>(initial, increment), node.lb[0])
            : std::min(adjust_resource<D>(initial, increment), node.ub[0]);
    return check_bounds<D>(new_resource, node.lb[0], node.ub[0]);
}

template <Direction D>
static bool process_non_disposable(double& new_resource, double initial,
                                   double increment, const VRPNode& node,
                                   const BucketOptions& opts) {
    new_resource = adjust_resource<D>(initial, increment);
    return check_bounds<D>(new_resource, node.lb[0], node.ub[0]);
}

template <Direction D>
static bool process_binary(double& new_resource, double initial,
                           double increment, const VRPNode& node,
                           const BucketOptions& opts) {
    new_resource = get_binary_result<D>(increment);
    return true;
}

template <Direction D>
static bool process_mtw(double& new_resource, double initial, double increment,
                        const VRPNode& node, const BucketOptions& opts) {
    if constexpr (D == Direction::Forward) {
        const auto adjusted = adjust_resource<D>(initial, increment);
        for (size_t i = 0; i < node.mtw_lb.size(); ++i) {
            new_resource = std::max(adjusted, node.mtw_lb[i]);
            if (check_bounds<D>(new_resource, node.lb[i], node.ub[i]))
                return true;
        }
    } else {
        const auto adjusted = adjust_resource<D>(initial, increment);
        for (size_t i = 0; i < node.mtw_ub.size(); ++i) {
            new_resource = std::min(adjusted, node.mtw_ub[i]);
            if (check_bounds<D>(new_resource, node.lb[i], node.ub[i]))
                return true;
        }
    }
    return false;
}

template <Direction D>
static bool process_battery(double& new_resource, double initial,
                            double increment, const VRPNode& node,
                            const BucketOptions& opts) {
    if (node.is_station) {
        new_resource = opts.battery_capacity;
        return true;
    }
    new_resource = initial;
    return D == Direction::Forward ? new_resource >= 0
                                   : new_resource <= opts.battery_capacity;
}

template <Direction D>
static bool process_recharge_count(double& new_resource, double initial,
                                   double increment, const VRPNode& node,
                                   const BucketOptions& opts) {
    if (!node.is_station) {
        new_resource = initial;
        return true;
    }
    new_resource = initial + 1;
    return new_resource <= opts.max_recharges;
}

template <Direction D>
static bool process_recharge_time(double& new_resource, double initial,
                                  double increment, const VRPNode& node,
                                  const BucketOptions& opts) {
    new_resource = node.is_station ? 0 : initial;
    return new_resource <= opts.max_recharge_time;
}

template <Direction D, typename Gamma, typename VRPNode>
bool BucketGraph::process_all_resources(
    std::vector<double>& new_resources,
    const std::array<double, R_SIZE>& initial_resources, const Gamma& gamma,
    const VRPNode& theNode, size_t N) {
    static constexpr ResourceProcessor processors[] = {
        &process_non_disposable<D>, &process_disposable<D>,
        &process_binary<D>,         &process_mtw<D>,
        &process_battery<D>,        &process_recharge_count<D>,
        &process_recharge_time<D>};

    for (size_t i = 0; i < N; ++i) {
        if (!processors[options.resource_type[i]](
                new_resources[i], initial_resources[i],
                gamma.resource_increment[i], theNode, options)) {
            return false;
        }
    }
    return true;
}
