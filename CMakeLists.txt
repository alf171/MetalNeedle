cmake_minimum_required(VERSION 3.28)

project(MetalNeedleBackend LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(METALNEEDLE_ENABLE_ASAN "Enable AddressSanitizer for backend builds" OFF)

find_package(Python3 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp" CACHE PATH "Homebrew libomp prefix")

pybind11_add_module(
  backend
  src/backend/bind.cc
  src/backend/cpu/backend.cc
  src/backend/metal/backend.cc
)

target_compile_features(backend PRIVATE cxx_std_23)

target_compile_options(
  backend
  PRIVATE
    -g
    -arch arm64
    -ftree-vectorize
    -funroll-loops
    -Xpreprocessor
    -fopenmp
    -Wall
    -frtti
    -ferror-limit=1000
    -stdlib=libc++
    -fno-objc-arc
)

if(METALNEEDLE_ENABLE_ASAN)
  target_compile_options(
    backend
    PRIVATE
      -fsanitize=address
      -fno-omit-frame-pointer
  )
  target_link_options(
    backend
    PRIVATE
      -fsanitize=address
  )
endif()

target_include_directories(
  backend
  PRIVATE
    "${LIBOMP_PREFIX}/include"
    src/backend/cpu
    metal-cpp
)

target_link_libraries(
  backend
  PRIVATE
    "${LIBOMP_PREFIX}/lib/libomp.dylib"
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
)

set_target_properties(
  backend
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tmp"
)
