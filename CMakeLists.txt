cmake_minimum_required(VERSION 3.29)
project(conv_lib)

set(CMAKE_CXX_STANDARD 20)

set(opencv_path "/home/perminov/clion/opencv/lib/cmake/opencv4")
find_package(OpenCV 4.9.0 EXACT REQUIRED PATHS ${opencv_path} NO_DEFAULT_PATH)
find_package(Boost COMPONENTS filesystem system)

message("-- Boost filesystem ${Boost_FILESYSTEM_LIBRARY}")
message("-- Boost system ${Boost_SYSTEM_LIBRARY}")
message("-- Boost includes ${Boost_INCLUDE_DIRS}")


add_library(conv_lib SHARED
        src/simple_conv.cpp
        src/simple_conv.h)

target_include_directories(conv_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
        conv_lib
        PUBLIC
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${OpenCV_LIBS}
)
