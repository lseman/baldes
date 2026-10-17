/* Autotools stand-in for Clp 1.17.10 and CoinUtils 2.11.
 * Selected by HAVE_CONFIG_H together with COINUTILS_BUILD and CLP_BUILD.
 * Third-party solvers (BLAS, LAPACK, MUMPS, ABC, ASL) stay off. */
#pragma once

#define PACKAGE "clp"
#define PACKAGE_NAME "Clp"
#define PACKAGE_VERSION "1.17.10"
#define VERSION "1.17.10"
#define STDC_HEADERS 1

#define HAVE_CASSERT 1
#define HAVE_CCTYPE 1
#define HAVE_CFLOAT 1
#define HAVE_CMATH 1
#define HAVE_CSTDARG 1
#define HAVE_CSTDDEF 1
#define HAVE_CSTDINT 1
#define HAVE_CSTDIO 1
#define HAVE_CSTDLIB 1
#define HAVE_CSTRING 1
#define HAVE_CTIME 1
#define HAVE_DLFCN_H 1
#define HAVE_FLOAT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_MATH_H 1
#define HAVE_MEMORY_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define HAVE_DRAND48 1
#define HAVE_RAND 1

#define SIZEOF_DOUBLE 8
#define SIZEOF_INT 4
#define SIZEOF_INT_P 8
#define SIZEOF_LONG __SIZEOF_LONG__
#define SIZEOF_LONG_LONG 8

#define COINUTILS_VERSION "2.11"
#define COINUTILS_VERSION_MAJOR 2
#define COINUTILS_VERSION_MINOR 11
#define COINUTILS_VERSION_RELEASE 11
#define COINUTILS_HAS_CSTDINT 1
#define COINUTILS_HAS_STDINT_H 1
#define COIN_COINUTILS_CHECKLEVEL 0
#define COIN_COINUTILS_VERBOSITY 0
#define COIN_INT64_T long long
#define COIN_UINT64_T unsigned long long
#define COIN_INTPTR_T long
#define COIN_C_FINITE std::isfinite
#define COIN_C_ISNAN std::isnan

#define CLP_VERSION "1.17.10"
#define CLP_VERSION_MAJOR 1
#define CLP_VERSION_MINOR 17
#define CLP_VERSION_RELEASE 10
#define COIN_CLP_CHECKLEVEL 0
#define COIN_CLP_VERBOSITY 0
#define COIN_HAS_CLP 1
#define COIN_HAS_COINUTILS 1

#define F77_FUNC(name, NAME) name##_
#define F77_FUNC_(name, NAME) name##__
