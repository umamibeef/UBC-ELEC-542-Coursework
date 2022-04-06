# set CMake minimum version required
cmake_minimum_required(VERSION 3.16)

# tell CMake where to find nvcc for CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set the program name
project(ExactHartreeFockSim LANGUAGES C CXX CUDA)

# GSL
find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
link_libraries(Eigen3::Eigen)

# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.71.0 COMPONENTS format program_options chrono)
link_libraries(${Boost_LIBRARIES})

# need to point to CUDA and Boost include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})

# add the executable
add_executable(ExactHartreeFockSim main.cpp randomize.cpp kernel.cu kernel.cu)

set_target_properties(ExactHartreeFockSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add compile options to enable GSL library
add_compile_options(`gsl-config --cflags --libs`)