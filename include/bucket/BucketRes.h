
#pragma once
#include "BucketGraph.h"
#include "Common.h"
#include "Definitions.h"

// Resource type enumeration for better type safety and readability
enum class ResourceType {
    Disposable = 1,
    NonDisposable = 0,
    Binary = 2,
    MTW = 3,
    Battery = 4,
    RechargeCount = 5,
    RechargeTime = 6
};

// Forward declare the BucketGraph class
class BucketGraph;

// Declare all specialized processing functions first
template <Direction D>
inline bool process_disposable_resource(double& new_resource,
                                        double initial_resource,
                                        double increment, double lb,
                                        double ub) {
    if constexpr (D == Direction::Forward) {
        new_resource = std::max(initial_resource + increment, lb);
        return new_resource <= ub;
    } else {
        new_resource = std::min(initial_resource - increment, ub);
        return new_resource >= lb;
    }
}

template <Direction D>
inline bool process_non_disposable_resource(double& new_resource,
                                            double initial_resource,
                                            double increment, double lb,
                                            double ub) {
    new_resource =
        initial_resource + (D == Direction::Forward ? increment : -increment);
    return new_resource >= lb && new_resource <= ub;
}

template <Direction D>
inline double process_binary_resource(double increment) {
    const bool is_forward = D == Direction::Forward;
    return (increment > 0) ? (is_forward ? 1.0 : 0.0)
                           : (is_forward ? 0.0 : 1.0);
}

template <Direction D>
inline bool process_battery_resource(double& new_resource,
                                     double initial_resource, bool is_station,
                                     double battery_capacity) {
    if (is_station) {
        new_resource = battery_capacity;
        return true;
    }
    new_resource = initial_resource;
    return D == Direction::Forward ? new_resource >= 0
                                   : new_resource <= battery_capacity;
}

template <Direction D>
inline bool process_recharge_count(double& new_resource,
                                   double initial_resource, bool is_station,
                                   double max_recharges) {
    if (!is_station) {
        new_resource = initial_resource;
        return true;
    }
    new_resource = initial_resource + 1;
    return new_resource <= max_recharges;
}

template <Direction D>
inline bool process_recharge_time(double& new_resource, double initial_resource,
                                  bool is_station, double max_recharge_time) {
    new_resource = is_station ? 0 : initial_resource;
    return new_resource <= max_recharge_time;
}

template <Direction D>
inline bool process_mtw_resource(double& new_resource, double initial_resource,
                                 double increment, const VRPNode& theNode) {
    if constexpr (D == Direction::Forward) {
        for (size_t i = 0; i < theNode.mtw_lb.size(); ++i) {
            new_resource =
                std::max(initial_resource + increment, theNode.mtw_lb[i]);
            if (new_resource <= theNode.ub[i]) return true;
        }
    } else {
        for (size_t i = 0; i < theNode.mtw_ub.size(); ++i) {
            new_resource =
                std::min(initial_resource - increment, theNode.mtw_ub[i]);
            if (new_resource >= theNode.lb[i]) return true;
        }
    }
    return false;
}

// Main processing function
template <Direction D, typename Gamma, typename VRPNode>
bool BucketGraph::process_all_resources(
    std::vector<double>& new_resources,
    const std::array<double, R_SIZE>& initial_resources, const Gamma& gamma,
    const VRPNode& theNode, size_t N) {
    for (size_t I = 0; I < N; ++I) {
        switch (static_cast<ResourceType>(options.resource_type[I])) {
            [[likely]] case ResourceType::Disposable:
                if (!process_disposable_resource<D>(
                        new_resources[I], initial_resources[I],
                        gamma.resource_increment[I], theNode.lb[I],
                        theNode.ub[I])) {
                    return false;
                }
                break;

            case ResourceType::NonDisposable:
                if (!process_non_disposable_resource<D>(
                        new_resources[I], initial_resources[I],
                        gamma.resource_increment[I], theNode.lb[I],
                        theNode.ub[I])) {
                    return false;
                }
                break;

            case ResourceType::Binary:
                new_resources[I] =
                    process_binary_resource<D>(gamma.resource_increment[I]);
                break;

            case ResourceType::Battery:
                if (!process_battery_resource<D>(
                        new_resources[I], initial_resources[I],
                        theNode.is_station, options.battery_capacity)) {
                    return false;
                }
                break;

            case ResourceType::RechargeCount:
                if (!process_recharge_count<D>(
                        new_resources[I], initial_resources[I],
                        theNode.is_station, options.max_recharges)) {
                    return false;
                }
                break;

            case ResourceType::RechargeTime:
                if (!process_recharge_time<D>(
                        new_resources[I], initial_resources[I],
                        theNode.is_station, options.max_recharge_time)) {
                    return false;
                }
                break;

            case ResourceType::MTW:
                if (!process_mtw_resource<D>(
                        new_resources[I], initial_resources[I],
                        gamma.resource_increment[I], theNode)) {
                    return false;
                }
                break;
        }
    }
    return true;
}
