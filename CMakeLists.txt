cmake_minimum_required(VERSION 3.10)
project(pqnt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

include_directories(include)

add_subdirectory(external/external/pybind11)

add_executable(pqnt_cli
    src/main.cpp
    src/pqnt.cpp
)

pybind11_add_module(pqnt
    src/bindings.cpp
    src/pqnt.cpp
)

