cmake_minimum_required(VERSION 3.10)
project(networks_minimal)

set(CMAKE_CXX_STANDARD 17)

# Eigen3
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

include_directories(${EIGEN3_INCLUDE_DIRS} include)
