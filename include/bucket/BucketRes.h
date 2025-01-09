
#pragma once
#include "BucketGraph.h"
#include "Common.h"
#include "Definitions.h"

template <Direction D, typename Gamma, typename VRPNode>
bool BucketGraph::process_all_resources(std::vector<double>              &new_resources,
                                  const std::array<double, R_SIZE> &initial_resources, const Gamma &gamma,
                                  const VRPNode &theNode, size_t N) {
    for (size_t I = 0; I < N; ++I) {
        if (!process_resource<D>(new_resources[I], initial_resources, gamma, theNode, I)) {
            return false; // baldesCtr violated, return false
        }
    }
    return true; // All resources processed successfully
}

constexpr int DISPOSABLE = 1;
constexpr int NON_DISPOSABLE = 0;
constexpr int BINARY = 2;
constexpr int MTW = 3;
constexpr int BATTERY_RESOURCE = 4;
constexpr int RECHARGE_COUNT = 5;
constexpr int RECHARGE_TIME = 6;

template <Direction D, typename Gamma, typename VRPNode>
constexpr bool BucketGraph::process_resource(double &new_resource, const std::array<double, R_SIZE> &initial_resources,
                                             const Gamma &gamma, const VRPNode &theNode, size_t I) {
    if (options.resource_type[I] == DISPOSABLE) { // Checked at compile-time
        if constexpr (D == Direction::Forward) {
            new_resource =
                std::max(initial_resources[I] + gamma.resource_increment[I], static_cast<double>(theNode.lb[I]));
            if (new_resource > theNode.ub[I]) {
                return false; // Exceeds upper bound, return false to stop processing
            }
        } else {
            new_resource =
                std::min(initial_resources[I] - gamma.resource_increment[I], static_cast<double>(theNode.ub[I]));
            if (new_resource < theNode.lb[I]) {
                return false; // Below lower bound, return false to stop processing
            }
        }
    } else if (options.resource_type[I] == NON_DISPOSABLE) {
        // TODO: Non-disposable resource handling, check if it is right
        if constexpr (D == Direction::Forward) {
            new_resource = initial_resources[I] + gamma.resource_increment[I];
            if (new_resource > theNode.ub[I]) {
                return false; // Exceeds upper bound, return false to stop processing
            } else if (new_resource < theNode.lb[I]) {
                return false; // Below lower bound, return false to stop processing
            }
        } else {
            new_resource = initial_resources[I] - gamma.resource_increment[I];
            if (new_resource > theNode.ub[I]) {
                return false; // Exceeds upper bound, return false to stop processing
            } else if (new_resource < theNode.lb[I]) {
                return false; // Below lower bound, return false to stop processing
            }
        }
    } else if (options.resource_type[I] == BINARY) {
        // TODO:: Binary resource handling, check if logic is right
        if constexpr (D == Direction::Forward) {
            // For binary resources, flip between 0 and 1 based on gamma.resource_increment[I]
            if (gamma.resource_increment[I] > 0) {
                new_resource = 1.0; // Switch "on"
            } else {
                new_resource = 0.0; // Switch "off"
            }
        } else {
            // In reverse, toggle as well
            if (gamma.resource_increment[I] > 0) {
                new_resource = 0.0; // Reverse logic: turn "off"
            } else {
                new_resource = 1.0; // Reverse logic: turn "on"
            }
        }
    } else if (options.resource_type[I] == MTW) {
        // TODO: handling multiple time windows case
        // "OR" resource case using mtw_lb and mtw_ub vectors for multiple time windows
        if constexpr (D == Direction::Forward) {
            bool is_feasible = false;
            for (size_t i = 0; i < theNode.mtw_lb.size(); ++i) {
                new_resource = std::max(initial_resources[I] + gamma.resource_increment[I], theNode.mtw_lb[i]);
                if (new_resource > theNode.ub[I]) {
                    continue; // Exceeds upper bound, try next time window
                } else {
                    is_feasible = true; // Feasible in this time window
                    break;
                }
            }

            if (!is_feasible) {
                return false; // Not feasible in any of the ranges
            }

            return true; // Successfully processed all resources
        } else {
            bool is_feasible = false;
            for (size_t i = 0; i < theNode.mtw_ub.size(); ++i) {
                new_resource = std::min(initial_resources[I] - gamma.resource_increment[I], theNode.mtw_ub[i]);
                if (new_resource < theNode.lb[I]) {
                    continue; // Below lower bound, try next time window }
                } else {
                    is_feasible = true; // Feasible in this time window break;
                }
            }

            if (!is_feasible) {
                return false; // Not feasible in any of the ranges
            }
        }
    } else if (options.resource_type[I] == BATTERY_RESOURCE) {
        if constexpr (D == Direction::Forward) {
            if (theNode.is_station) {
                new_resource = options.battery_capacity; // Full recharge
            } else {
                new_resource = initial_resources[I];// - gamma.energy_consumption;
            }
            if (new_resource < 0) return false;
        } else {
            // Backward direction analogous but reversed
            if (theNode.is_station) {
                new_resource = options.battery_capacity;
            } else {
                new_resource = initial_resources[I];// + gamma.energy_consumption;
            }
            if (new_resource > options.battery_capacity) return false;
        }
    }
    else if (options.resource_type[I] == RECHARGE_COUNT) {
        if (theNode.is_station) {
            new_resource = initial_resources[I] + 1;
            if (new_resource > options.max_recharges) return false;
        } else {
            new_resource = initial_resources[I];
        }
    }
    else if (options.resource_type[I] == RECHARGE_TIME) {
        if (theNode.is_station) {
            new_resource = 0;//gamma.recharge_time;
        } else {
            new_resource = initial_resources[I];// + gamma.recharge_time;
            if (new_resource > options.max_recharge_time) return false;
        }
    }
    return true; // Successfully processed all resources
}