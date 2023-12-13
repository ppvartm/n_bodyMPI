cmake_minimum_required (VERSION 3.11)

project (RungeKutta-MPI)
set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(RungeKutta-MPI main.cpp)

target_link_libraries(RungeKutta-MPI MPI::MPI_CXX)