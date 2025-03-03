cmake_minimum_required(VERSION 3.29)
project(MARLIP_1)

set(CMAKE_CXX_STANDARD 20)

add_executable(MARLIP_1
        cpp/data_generator.cpp
        cpp/knapsack.cpp
        cpp/ReadData.cpp
        cpp/dual_factor.cpp)
