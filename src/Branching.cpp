#include "Dual.h"
#include "bnb/Node.h"

void BranchingDuals::computeDuals(BNBNode *model, double threshold) {
    arcDuals_  = ArcDuals();
    nodeDuals_ = NodeDuals();

    // First pass: Compute dual values and store them
    for (int i = 0; i < branchingCandidates_.size(); ++i) {
        size_t size = std::min(branchingCandidates_.size(), branchingConstraints_.size());

        for (size_t i = 0; i < size; ++i) {
            auto candidate = branchingCandidates_[i];
            auto ctr       = branchingConstraints_[i];

            double dualValue = model->getDualVal(ctr->index());
            if (std::abs(dualValue) < threshold) { continue; }

            if (candidate->getCandidateType() == CandidateType::Edge) {
                RawArc arc(candidate->sourceNode, candidate->targetNode);
                arcDuals_.setOrIncrementDual(arc, dualValue);
            } else if (candidate->getCandidateType() == CandidateType::Node) {
                nodeDuals_.setOrIncrementDual(candidate->targetNode, dualValue);
            }
        }
    }
}