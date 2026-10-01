#include "bnb/bcp/PricingState.h"

#include <cassert>

using baldes::bcp::PricingState;

int main() {
    assert(!PricingState::canApplyDeluxing(false, false));
    assert(PricingState::canApplyDeluxing(true, false));

    // A capped enumeration is not a proof that every route inside the
    // incumbent gap was generated.  Reduced-cost fixing would be unsafe.
    assert(!PricingState::canApplyDeluxing(true, true));

    assert(PricingState::hasSrcMasterCapacity(49, 50));
    assert(!PricingState::hasSrcMasterCapacity(50, 50));
    assert(!PricingState::hasSrcMasterCapacity(51, 50));
}
