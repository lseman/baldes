cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Simplex3 VERSION 1.0 LANGUAGES C CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Eigen3 library
find_package(Eigen3 3.3 REQUIRED)

# Add the executable
add_executable(lu lu.cpp)

# Link Eigen to the executable
target_link_libraries(lu PRIVATE Eigen3::Eigen)

# Include Eigen directories (this is handled by `find_package`, can be omitted)
# target_include_directories(lu PRIVATE ${EIGEN3_INCLUDE_DIRS})

# Find all .c files in the basiclu directory
file(GLOB BASICLU_SOURCES /data/solver/bcp/bucket/third_party/basiclu/*.c)

# Print the source files (for debugging purposes)
message(STATUS "BASICLU_SOURCES: ${BASICLU_SOURCES}")

# Add the basiclu source files to the target
target_sources(lu PRIVATE ${BASICLU_SOURCES})

# Include the basiclu header directory
target_include_directories(lu PRIVATE /data/solver/bcp/bucket/third_party/basiclu)

# Find fmt library and link it
find_package(fmt REQUIRED)
target_link_libraries(lu PRIVATE fmt::fmt)
