#pragma once

#include "Definitions.h"
#include "bnb/Node.h"
#include <memory>

class BNBNode;
/**
 * @brief The Problem class represents an abstract base class for defining optimization problems.
 *
 * This class provides the interface for branching, bounding, and calculating the objective value
 * for a given node in the optimization problem. Derived classes must implement these methods
 * based on the specific problem's requirements.
 */
class Problem {
public:
    Problem() = default;

    /**
     * @brief Branches the given node.
     *
     * @param node A shared pointer to the Node object.
     */
    virtual void branch(BNBNode *node) {};

    /**
     * @brief Calculates the bound for the given node.
     *
     * @param node A shared pointer to the Node object.
     * @return double The bound for the given node.
     */
    virtual void evaluate(BNBNode *node) {};

    /**
     * @brief Calculates the objective value for the given node.
     *
     * @param node A shared pointer to the Node object.
     * @return double The objective value for the given node.
     */
    virtual double objective(BNBNode *node) = 0;

    virtual bool CG(BNBNode *node, int max_iter = 2000) { return false; }

    virtual double bound(BNBNode *node) = 0;

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~Problem() = default;
};
