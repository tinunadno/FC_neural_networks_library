cmake_minimum_required(VERSION 3.29)
project(conv_lib)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_library(conv_lib SHARED
        src/simple_conv.cpp
        src/simple_conv.h)

target_include_directories(conv_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
        conv_lib PRIVATE
        ${OpenCV_LIBS}
        Boost::filesystem
)
