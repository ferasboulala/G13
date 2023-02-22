cmake_minimum_required(VERSION 3.0)

project(G13)

find_library(METAL Metal)
find_library(FOUNDATION Foundation)
find_library(METAL_KIT MetalKit)

add_compile_options(
    -Ofast
    -march=native
    -Wall
    -Wno-pedantic
    -Werror
    -std=c++17
    -std=gnu++17
    -stdlib=libc++
    -fno-objc-arc
    -I./metal-cpp
    -fverbose-asm
    -save-temps)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ALL_LIBRARIES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin/")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")

include_directories(metal-cpp)

add_subdirectory(src)
add_subdirectory(apps)
