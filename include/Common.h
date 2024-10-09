/*
 * @file Common.h
 * @brief Header file containing common includes and definitions used across the project.
 *
 * This file contains common includes and definitions used across the project, such as standard library headers,
 * definitions for constants, and utility functions.
 */
#pragma once
#include "config.h"

#include "gurobi_c++.h"
#include "gurobi_c.h"

#include "../third_party/pdqsort.h"

#include "Hashes.h"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <execution>
#ifdef AVX
#include <immintrin.h>
#endif

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

#include "ankerl/unordered_dense.h"
#include <fmt/color.h>
#include <fmt/core.h>