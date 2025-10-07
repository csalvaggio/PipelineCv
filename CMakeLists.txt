cmake_minimum_required(VERSION 3.16)

project(pcv_pipeline LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Place executables in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Show what was found (optional but useful)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Add include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(test_pipeline_cv test_pipeline_cv.cpp)

# Link against OpenCV
target_link_libraries(test_pipeline_cv PRIVATE ${OpenCV_LIBS})

