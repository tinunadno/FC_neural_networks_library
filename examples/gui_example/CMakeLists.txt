cmake_minimum_required(VERSION 3.29)
project(conv_lib_gui_example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_subdirectory(../../ lib_conv_build)

add_compile_definitions(-DCONV_HOME="${PROJECT_SOURCE_DIR}/../")

add_executable(conv_lib_gui_example main.cpp MainWindow.cpp MainWindow.h)

include_directories(${PROJECT_SOURCE_DIR}/../..)

target_link_libraries(conv_lib_gui_example PRIVATE
        conv_lib
        Qt5::Widgets
)