/*
 * @file State.h
 * @brief Declares State interfaces and types used by the BALDES solver.
 *
 * This file declares the State interfaces and helper functions used by the BALDES solver.
 *
 */


#pragma once

#include "Cut.h"
#include "Path.h"
#include "Serializer.h"

struct Snapshot {
    Snapshot() = default;
    std::vector<Path> paths;
    CutStorage        cutStorage;

    REFLECT(paths, cutStorage);
};
