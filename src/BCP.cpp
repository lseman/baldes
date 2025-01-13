#include "bnb/BCP.h"

#include "bnb/Branching.h"

std::unique_ptr<VRProblem> VRProblem::clone() const {
    auto newProblem = std::make_unique<VRProblem>();
    newProblem->instance = instance;
    newProblem->nodes = nodes;
    return newProblem;
}

void VRProblem::branch(BNBNode *node) {
    fmt::print("\033[34m_STARTING BRANCH PROCEDURE \033[0m");
    fmt::print("\n");

    node->relaxNode();

    auto candidates = Branching::VRPStandardBranching(node, &instance, this);
    auto candidateCounter = 0;

    for (auto candidate : candidates) {
        if (node->hasCandidate(candidate)) continue;
        if (node->hasRaisedChild(candidate)) continue;

        auto candidatosNode = node->getCandidatos();
        // print len candidatosNode
        auto childNode = node->newChild();
        childNode->addCandidate(candidate);
        node->addRaisedChildren(candidate);
        node->addChildren(childNode);

        candidateCounter++;
        if (candidateCounter >= NUMERO_CANDIDATOS) break;
    }

    fmt::print("\033[34m_FINISHED BRANCH PROCEDURE \033[0m");
    fmt::print("\n");
}
