/*
 * @file Common.h
 * @brief Header file containing common includes and definitions used across the project.
 *
 * This file contains common includes and definitions used across the project, such as standard library headers,
 * definitions for constants, and utility functions.
 */
#pragma once
#include "config.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include "../third_party/pdqsort.h"

#include "Hashes.h"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <execution>


#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <vector>
#include <queue>
#include <string_view>

#include "ankerl/unordered_dense.h"

#include "../third_party/small_vector.hpp"

#include <fmt/color.h>
#include <fmt/core.h>

#include "xxhash.h" // Include the header file for xxhash
#include <stack>
