cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
# 添加opencv库
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_imgcodecs)