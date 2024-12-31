cmake_minimum_required(VERSION 3.14)
project(SupernodalTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (ensure Eigen is installed or provide the path)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIRS} include)

# Add executable
add_executable(SupernodalTest main.cpp)

# Link Eigen (not required for header-only, but good practice)
target_link_libraries(SupernodalTest Eigen3::Eigen)