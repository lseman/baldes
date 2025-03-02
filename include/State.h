
#pragma once

#include "Cut.h"
#include "Path.h"
#include "Serializer.h"

struct Snapshot {
    Snapshot() = default;
    std::vector<Path> paths;
    CutStorage cutStorage;

    REFLECT(paths, cutStorage);
};
