cmake_minimum_required(VERSION 3.28)
project(conv_lib)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(MKL_NUM_THREADS 4)

#set(MKL_DIR "/home/yura/intel/oneapi/mkl/latest/lib/cmake/mkl/")
#set(MKL_INCLUDE_DIRS "/home/yura/intel/oneapi/mkl/latest/include/")
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system)
find_package(MKL REQUIRED)

set(BUILD_TYPE STATIC)

if(${BUILD_SHARED_LIBS})
    set(BUILD_TYPE SHARED)
endif(${BUILD_SHARED_LIBS})

message("-- Boost filesystem ${Boost_FILESYSTEM_LIBRARY}")
message("-- Boost system ${Boost_SYSTEM_LIBRARY}")
message("-- Boost includes ${Boost_INCLUDE_DIRS}")
message("-- MKL includes ${MKL_INCLUDE_DIRS}")
message("-- MKL root ${MKL_ROOT}")


add_library(conv_lib ${BUILD_TYPE}
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
        ${MKL_ROOT}/lib/libmkl_intel_thread.so
        ${MKL_ROOT}/../../compiler/latest/lib/libiomp5.so
        ${MKL_ROOT}/lib/libmkl_core.so
        ${MKL_ROOT}/lib/libmkl_blacs_openmpi_ilp64.so
        pthread
        m
        dl
)
