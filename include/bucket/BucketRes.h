#pragma once
#include "BucketGraph.h"
#include "Common.h"
#include "Definitions.h"

// Keep original constexpr definitions as static members
static constexpr int DISPOSABLE = 1;
static constexpr int NON_DISPOSABLE = 0;
static constexpr int BINARY = 2;
static constexpr int MTW = 3;
static constexpr int BATTERY_RESOURCE = 4;
static constexpr int RECHARGE_COUNT = 5;
static constexpr int RECHARGE_TIME = 6;

namespace {
// Helper functions for each resource type - moved into anonymous namespace
template <Direction D>
[[nodiscard]] constexpr bool process_disposable_resource(
    double& new_resource,
    const double initial_resource,
    const double increment,
    const auto& theNode,
    const size_t I) {
    
    if constexpr (D == Direction::Forward) {
        new_resource = std::max(initial_resource + increment, 
                              static_cast<double>(theNode.lb[I]));
        return new_resource <= theNode.ub[I];
    } else {
        new_resource = std::min(initial_resource - increment, 
                              static_cast<double>(theNode.ub[I]));
        return new_resource >= theNode.lb[I];
    }
}

template <Direction D>
[[nodiscard]] constexpr bool process_non_disposable_resource(
    double& new_resource,
    const double initial_resource,
    const double increment,
    const auto& theNode,
    const size_t I) {
    
    new_resource = initial_resource + (D == Direction::Forward ? increment : -increment);
    return new_resource >= theNode.lb[I] && new_resource <= theNode.ub[I];
}

template <Direction D>
[[nodiscard]] constexpr bool process_binary_resource(
    double& new_resource,
    const double increment) {
    
    new_resource = (increment > 0) ^ (D == Direction::Backward) ? 1.0 : 0.0;
    return true;
}

template <Direction D>
[[nodiscard]] bool process_mtw_resource(
    double& new_resource,
    const double initial_resource,
    const double increment,
    const auto& theNode,
    const size_t I) {  // Added I parameter
    
    if constexpr (D == Direction::Forward) {
        return std::any_of(theNode.mtw_lb.begin(), theNode.mtw_lb.end(),
            [&, I](const auto& lb) {  // Capture I explicitly
                new_resource = std::max(initial_resource + increment, lb);
                return new_resource <= theNode.ub[I];
            });
    } else {
        return std::any_of(theNode.mtw_ub.begin(), theNode.mtw_ub.end(),
            [&, I](const auto& ub) {  // Capture I explicitly
                new_resource = std::min(initial_resource - increment, ub);
                return new_resource >= theNode.lb[I];
            });
    }
}

template <Direction D>
[[nodiscard]] constexpr bool process_battery_resource(
    double& new_resource,
    const double initial_resource,
    const auto& gamma,
    const auto& theNode,
    const auto& options) {  // Added options parameter
    
    if (theNode.is_station) {
        new_resource = options.battery_capacity;
        return true;
    }
    
    new_resource = initial_resource;
    if constexpr (D == Direction::Forward) {
        return new_resource >= 0;
    } else {
        return new_resource <= options.battery_capacity;
    }
}

template <Direction D>
[[nodiscard]] constexpr bool process_recharge_count(
    double& new_resource,
    const double initial_resource,
    const auto& theNode,
    const auto& options) {  // Added options parameter
    
    if (theNode.is_station) {
        new_resource = initial_resource + 1;
        return new_resource <= options.max_recharges;
    }
    new_resource = initial_resource;
    return true;
}

template <Direction D>
[[nodiscard]] constexpr bool process_recharge_time(
    double& new_resource,
    const double initial_resource,
    const auto& gamma,
    const auto& theNode,
    const auto& options) {  // Added options parameter
    
    if (theNode.is_station) {
        new_resource = 0;
        return true;
    }
    new_resource = initial_resource;
    return new_resource <= options.max_recharge_time;
}
} // namespace

template <Direction D, typename Gamma, typename VRPNode>
bool BucketGraph::process_all_resources(
    std::vector<double>& new_resources,
    const std::array<double, R_SIZE>& initial_resources,
    const Gamma& gamma,
    const VRPNode& theNode,
    size_t N) {
    
    return std::all_of(
        new_resources.begin(),
        new_resources.begin() + N,
        [&, resource_idx = size_t(0)](double& resource) mutable {
            return this->process_resource<D>(
                resource,
                initial_resources,
                gamma,
                theNode,
                resource_idx++
            );
        });
}

template <Direction D, typename Gamma, typename VRPNode>
constexpr bool BucketGraph::process_resource(
    double& new_resource,
    const std::array<double, R_SIZE>& initial_resources,
    const Gamma& gamma,
    const VRPNode& theNode,
    const size_t I) {

    switch (this->options.resource_type[I]) {
        case DISPOSABLE:
            return process_disposable_resource<D>(
                new_resource, initial_resources[I], gamma.resource_increment[I], theNode, I);

        case NON_DISPOSABLE:
            return process_non_disposable_resource<D>(
                new_resource, initial_resources[I], gamma.resource_increment[I], theNode, I);

        case BINARY:
            return process_binary_resource<D>(
                new_resource, gamma.resource_increment[I]);

        case MTW:
            return process_mtw_resource<D>(
                new_resource, initial_resources[I], gamma.resource_increment[I], theNode, I);

        case BATTERY_RESOURCE:
            return process_battery_resource<D>(
                new_resource, initial_resources[I], gamma, theNode, this->options);

        case RECHARGE_COUNT:
            return process_recharge_count<D>(
                new_resource, initial_resources[I], theNode, this->options);

        case RECHARGE_TIME:
            return process_recharge_time<D>(
                new_resource, initial_resources[I], gamma, theNode, this->options);

        default:
            return false; // Invalid resource type
    }
}