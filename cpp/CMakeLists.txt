cmake_minimum_required(VERSION 3.10)

project(MultiAgentReinforcement)

set(CMAKE_CXX_STANDARD 14)

# 创建共享库
add_library(dynamic_planner SHARED
        data_generator.cpp
        knapsack.cpp
)

# 设置C++标准和编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
