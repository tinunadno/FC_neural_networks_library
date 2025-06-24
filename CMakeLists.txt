cmake_minimum_required(VERSION 3.29)
project(conv_lib)

set(CMAKE_CXX_STANDARD 20)

set(opencv_path "/home/perminov/clion/opencv/lib/cmake/opencv4")
find_package(OpenCV 4.9.0 EXACT REQUIRED PATHS ${opencv_path} NO_DEFAULT_PATH)
find_package(Boost COMPONENTS filesystem system)
set(MKL_DIR "/home/perminov/intel/oneapi/mkl/latest/")
find_package(MKL REQUIRED)
set(MKL_INCLUDE_DIRS "/home/perminov/intel/oneapi/mkl/latest/include/")

message("-- Boost filesystem ${Boost_FILESYSTEM_LIBRARY}")
message("-- Boost system ${Boost_SYSTEM_LIBRARY}")
message("-- Boost includes ${Boost_INCLUDE_DIRS}")
message("-- MKL includes ${MKL_INCLUDE_DIRS}")
message("-- MKL root ${MKL_ROOT}")


add_library(conv_lib SHARED
        src/simple_conv.cpp
        src/simple_conv.h
        src/blas_impl.cpp
        src/blas_impl.h
        src/io.cpp
        src/learning.cpp
        src/private.cpp
        src/sc_private.h
        src/preprocessing.cpp
)


target_include_directories(conv_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${MKL_INCLUDE_DIRS}
)

target_link_libraries(
        conv_lib
        PUBLIC
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${OpenCV_LIBS}
        ${MKL_ROOT}/lib/libmkl_intel_lp64.so
        ${MKL_ROOT}/lib/libmkl_sequential.so
        ${MKL_ROOT}/lib/libmkl_core.so
)
