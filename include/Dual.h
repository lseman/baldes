#pragma once

#include "Arc.h"
#include "Hashes.h"
#include "MIPHandler/Constraint.h"
#include "MIPHandler/MIPHandler.h"
#include "ankerl/unordered_dense.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "VRPCandidate.h"

// Forward declare NodeDuals class
class NodeDuals;

// Structure to store and manage dual values for arcs
class ArcDuals {
public:
    // Add or update the dual value for an arc
    void setDual(const RawArc &arc, double dualValue) { arcDuals_[arc] = dualValue; }

    // Retrieve the dual value for an arc; returns 0 if the arc does not have a dual
    double getDual(int i, int j) const {
        RawArc arc(i, j);
        auto   it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            return it->second; // Return the dual if found
        }
        return 0.0; // Default to 0 if not found
    }

    double getDual(RawArc arc) const {
        auto it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            return it->second; // Return the dual if found
        }
        return 0.0; // Default to 0 if not found
    }

    void setOrIncrementDual(const RawArc &arc, double dualValue) {
        auto it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            it->second += dualValue; // Increment the dual if the arc already has a dual
        } else {
            arcDuals_[arc] = dualValue; // Set the dual if the arc does not have a dual
        }
    }

private:
    ankerl::unordered_dense::map<RawArc, double, RawArcHash> arcDuals_; // Map for storing arc duals
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

    void addCandidate(VRPCandidate *candidate, Constraint &constraint) {
        branchingCandidates_.push_back(candidate);
        branchingConstraints_.push_back(constraint);
    }

    void setDual(CandidateType type, const RawArc &arc, double dualValue) {
        if (type == CandidateType::Edge) { arcDuals_.setDual(arc, dualValue); }
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

    void computeDuals(MIPProblem *model, double threshold = 1e-3) {
        arcDuals_  = ArcDuals();
        nodeDuals_ = NodeDuals();

        // First pass: Compute dual values and store them
        for (int i = 0; i < branchingCandidates_.size(); ++i) {
            const auto &candidate = branchingCandidates_[i];
            // TODO: Check if the constraint is violated
            double dualValue = 0; // branchingConstraints_[i].get(GRB_DoubleAttr_Pi);
            if (std::abs(dualValue) < threshold) { continue; }

            if (candidate->getCandidateType() == CandidateType::Edge) {
                RawArc arc(candidate->sourceNode, candidate->targetNode);
                arcDuals_.setOrIncrementDual(arc, dualValue);
            } else if (candidate->getCandidateType() == CandidateType::Node) {
                nodeDuals_.setOrIncrementDual(candidate->targetNode, dualValue);
            }
        }
    }

    // define size as size of branchingCandidates_
    int size() { return branchingCandidates_.size(); }

private:
    std::vector<VRPCandidate *> branchingCandidates_;
    std::vector<Constraint>     branchingConstraints_;
    ArcDuals                    arcDuals_;
    NodeDuals                   nodeDuals_;
};
