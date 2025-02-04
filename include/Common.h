/*
 * @file Common.h
 * @brief Header file containing common includes and definitions used across the
 * project.
 *
 * This file contains common includes and definitions used across the project,
 * such as standard library headers, definitions for constants, and utility
 * functions.
 */
#pragma once
#include "config.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include <fmt/color.h>
#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <exec/static_thread_pool.hpp>
#include <execution>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <ranges>
#include <stack>
#include <stdexec/execution.hpp>
#include <string_view>
#include <vector>

#include "../third_party/pdqsort.h"
#include "../third_party/small_vector.hpp"
#include "Hashes.h"
#include "ankerl/unordered_dense.h"
#include "xxhash.h"  // Include the header file for xxhash
