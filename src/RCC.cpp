#include "RCC.h"
#include "Cut.h"
#include "bnb/Node.h"

ArcDuals RCCManager::computeDuals(BNBNode *model, double threshold) {
    ArcDuals arcDuals;
    // First pass: Compute dual values and store them
    for (int i = 0; i < cuts_.size(); ++i) {
        const auto &cut = cuts_[i];

        // TODO: adjust to new stuff
        double dualValue = model->getDualVal(cut.ctr->index());

        if (std::abs(dualValue) < 1e-3) { continue; }

        // Sum the dual values for all arcs in this cut
        for (const auto &arc : cut.arcs) {
            arcDuals.setOrIncrementDual(arc, dualValue); // Update arc duals
        }
    }

    // Second pass: Remove cuts with dual values near zero
    /*
    cuts_.erase(std::remove_if(cuts_.begin(), cuts_.end(),
                               [&](const RCCut &cut, size_t i = 0) mutable {
                                   if (std::abs(dualValues[i]) < threshold) {
                                       model->remove(cut.ctr); // Remove the constraint from the model
                                       return true;            // Mark this cut for removal
                                   }
                                   ++i;
                                   return false; // Keep this cut
                               }),
                cuts_.end());
*/
    return arcDuals;
}