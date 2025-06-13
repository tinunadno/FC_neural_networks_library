cmake_minimum_required(VERSION 3.29)
project(conv_lib_usage_example)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_subdirectory(../../ lib_conv_build)

add_compile_definitions(-DCONV_HOME="${PROJECT_SOURCE_DIR}/../")

add_executable(conv_lib_usage_example main.cpp)

include_directories(${PROJECT_SOURCE_DIR}/../..)

target_link_libraries(conv_lib_usage_example PRIVATE
        conv_lib
        ${OpenCV_LIBS}
)