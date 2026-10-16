#ifndef CONFIG_H
#define CONFIG_H

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// Enable or disable specific features based on CMake options
/* #undef RIH */
#define RCC
#define SRC
/* #undef UNREACHABLE_DOMINANCE */
#define SORTED_LABELS
#define INTRUSIVE_BUCKET_LABELS
/* #undef MCD */
#define FIX_BUCKETS
#define IPM
/* #undef TR */
/* #undef STAB */
/* #undef WITH_PYTHON */
/* #undef EVRP */
/* #undef MTW */
/* #undef SCHRODINGER */
#define JEMALLOC
/* #undef GUROBI */
#define HIGHS
/* #undef NSYNC */
/* #undef IPM_ACEL */
/* #undef CUSTOM_COST */
/* #undef ITERATIVE_HGS */

// Define constants for resource sizes
#define R_SIZE 1
#define N_SIZE 102
#define MAX_SRC_CUTS 50
#define MAX_SRC_CUTS_PER_ROUND 20
#define MAX_RANK3_SRC_CUTS_PER_ROUND 8
#define BUCKET_CAPACITY 100
#define TIME_INDEX 0
#define DEMAND_INDEX 
#define N_ADD 10
#define HGS_TIME 2
#define VERBOSE_LEVEL 0
#endif  // CONFIG_H
