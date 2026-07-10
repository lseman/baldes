/*
 * @file State.h
 * @brief Declares State interfaces and types used by the BALDES solver.
 *
 * This file declares the State interfaces and helper functions used by the BALDES solver.
 *
 */


#pragma once

#include "cuts/Cut.h"
#include "model/Path.h"
#include "routing/Serializer.h"

struct Snapshot {
    Snapshot() = default;
    std::vector<Path> paths;
    CutStorage        cutStorage;

    REFLECT(paths, cutStorage);
};
