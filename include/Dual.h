#pragma once

#include "Common.h"

#include "Arc.h"
#include "Hashes.h"

#include "miphandler/Constraint.h"
#include "miphandler/MIPHandler.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include "VRPCandidate.h"

// Forward declare NodeDuals class
class NodeDuals;

class BNBNode;

// Structure to store and manage dual values for arcs
class ArcDuals {
public:
    // Add or update the dual value for an arc
    void setDual(const RawArc& arc, double dualValue) {
        arcDuals_[arc] = dualValue;
    }

    // Retrieve the dual value for an arc
    double getDual(int i, int j) const {
        return getDual(RawArc(i, j));
    }

    double getDual(const RawArc& arc) const {
        auto it = arcDuals_.find(arc);
        return (it != arcDuals_.end()) ? it->second : 0.0;
    }

    // Set or increment dual value in one operation
    void setOrIncrementDual(const RawArc& arc, double dualValue) {
        arcDuals_[arc] += dualValue; // Will create with 0 + dualValue if not exists
    }

private:
    ankerl::unordered_dense::map<RawArc, double, RawArcHash> arcDuals_;
};

// Structure to store and manage dual values for nodes
class NodeDuals {
public:
    void setDual(int node, double dualValue) { nodeDuals_[node] = dualValue; }

    double getDual(int node) const {
        auto it = nodeDuals_.find(node);
        return (it != nodeDuals_.end()) ? it->second : 0.0;
    }

    void setOrIncrementDual(int node, double dualValue) {
        auto it = nodeDuals_.find(node);
        if (it != nodeDuals_.end()) {
            it->second += dualValue;
        } else {
            nodeDuals_[node] = dualValue;
        }
    }

private:
    ankerl::unordered_dense::map<int, double> nodeDuals_;
};

// Unified manager for branching duals
class BranchingDuals {
public:
    std::vector<VRPCandidate *> getBranchingCandidates() { return branchingCandidates_; }

    bool empty() { return branchingCandidates_.empty(); }

    void addCandidate(VRPCandidate *candidate, baldesCtrPtr constraint) {
        branchingCandidates_.push_back(candidate);
        branchingbaldesCtrs_.push_back(constraint);
    }

    void setDual(CandidateType type, const RawArc &arc, double dualValue) {
        if (type == CandidateType::Edge) { arcDuals_.setDual(arc, dualValue); }
    }

    std::vector<double> computeCoefficients(const std::vector<uint16_t> route) {
        std::vector<double> coefficients;
        for (auto candidate : branchingCandidates_) {
            if (candidate->getCandidateType() == CandidateType::Node) {
                bool has_node = false;
                for (auto node : route) {
                    if (candidate->targetNode == node) {
                        has_node = true;
                        break;
                    }
                }
                if (has_node) {
                    coefficients.push_back(1.0);
                } else {
                    coefficients.push_back(0.0);
                }
            } else if (candidate->getCandidateType() == CandidateType::Edge) {
                bool has_edge = false;
                for (size_t i = 1; i < route.size(); ++i) {
                    if (candidate->sourceNode == route[i - 1] && candidate->targetNode == route[i]) {
                        has_edge = true;
                        break;
                    }
                }
                if (has_edge) {
                    coefficients.push_back(1.0);
                } else {
                    coefficients.push_back(0.0);
                }
            } else if (candidate->getCandidateType() == CandidateType::Vehicle) {
                coefficients.push_back(1.0);
            } else if (candidate->getCandidateType() == CandidateType::Cluster) {
                bool has_cluster = false;
                for (auto node : route) {
                    for (auto cluster_node : candidate->cluster) {
                        if (node == cluster_node) {
                            has_cluster = true;
                            break;
                        }
                    }
                }
                if (has_cluster) {
                    coefficients.push_back(1.0);
                } else {
                    coefficients.push_back(0.0);
                }
            } else {
                coefficients.push_back(0.0);
            }
        }
        return coefficients;
    }

    void setDual(CandidateType type, int node, double dualValue) {
        if (type == CandidateType::Node) { nodeDuals_.setDual(node, dualValue); }
    }

    double getDual(const RawArc &arc) const { return arcDuals_.getDual(arc); }

    double getDual(int i, int j) const {
        RawArc arc(i, j);
        return arcDuals_.getDual(arc);
    }

    double getDual(int node) const { return nodeDuals_.getDual(node); }

    void computeDuals(BNBNode *model, double threshold = 1e-3);

    auto getBranchingbaldesCtrs() { return branchingbaldesCtrs_; }

    // define size as size of branchingCandidates_
    int size() { return branchingCandidates_.size(); }

private:
    std::vector<VRPCandidate *> branchingCandidates_;
    std::vector<baldesCtrPtr>   branchingbaldesCtrs_;
    ArcDuals                    arcDuals_;
    NodeDuals                   nodeDuals_;
};
