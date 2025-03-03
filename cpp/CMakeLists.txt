cmake_minimum_required(VERSION 3.10)

project(MultiAgentReinforcement)

set(CMAKE_CXX_STANDARD 14)

# 创建共享库
add_library(dynamic_planner SHARED
        data_generator.cpp
        knapsack.cpp
        dual_factor.cpp
)

# 设置C++标准和编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# 设置输出的共享库文件名
set_target_properties(data_generator knapsack PROPERTIES OUTPUT_NAME "libdata_generator" "libknapsack")