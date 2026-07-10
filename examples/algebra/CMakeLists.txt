cmake_minimum_required(VERSION 3.14)
project(SupernodalTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (ensure Eigen is installed or provide the path)
find_package(Eigen3 REQUIRED)

# Add executable
add_executable(SupernodalTest main.cpp)

target_include_directories(SupernodalTest PRIVATE ../../include)

# Link Eigen (not required for header-only, but good practice)
target_link_libraries(SupernodalTest PRIVATE Eigen3::Eigen)
