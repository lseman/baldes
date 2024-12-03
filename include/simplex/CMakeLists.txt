# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(ModernSimplex)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED)

# Add the executable
add_executable(ModernSimplex main.cpp)

# Link Eigen to the target
target_link_libraries(ModernSimplex PRIVATE Eigen3::Eigen)

# Enable all compiler warnings (optional but recommended during development)
if (MSVC)
    target_compile_options(ModernSimplex PRIVATE /W4)
else()
    target_compile_options(ModernSimplex PRIVATE -Wall -Wextra -pedantic)
endif()
