cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# 设置c++11编译
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 项目名称
project(demo)

# 拿到pytorch c++包
find_package(Torch REQUIRED)

# 输出项目名称,依赖main.cpp
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")