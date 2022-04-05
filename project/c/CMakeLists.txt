# set CMake minimum version required
cmake_minimum_required(VERSION 3.16)

# tell CMake where to find nvcc for CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set the program name
project(ExactHartreeFockSim LANGUAGES C CXX CUDA)

# need GSL and CUDA
find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)

# need to point to CUDA include dirs since we're including cuda_runtime.h in some files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# add the executable
add_executable(ExactHartreeFockSim main.c randomize.cpp kernel.cu kernel.cu)

set_target_properties(ExactHartreeFockSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add compile options to enable GSL library
add_compile_options(`gsl-config --cflags --libs`)