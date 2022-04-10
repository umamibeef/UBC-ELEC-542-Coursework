# set CMake minimum version required
cmake_minimum_required(VERSION 3.23)

# tell CMake where to find extra files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

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

# LAPACK/BLAS
find_package(BLAS)
find_package(LAPACK)
find_package(LAPACKE)
if(LAPACK_FOUND AND BLAS_FOUND AND LAPACKE_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
endif()
link_libraries(${lapackblas_libraries})

# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.71.0 COMPONENTS program_options date_time)
link_libraries(${Boost_LIBRARIES})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

# need to point to CUDA and Boost include dirs
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})

# add the executable
add_executable(ExactHartreeFockSim main.cpp console.cpp kernel.cu kernel.cu)

set_target_properties(ExactHartreeFockSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add compile options to enable GSL library
add_compile_options(`gsl-config --cflags --libs`)

# Getting git hash and branch into C++ as a define
# http://xit0.org/2013/04/cmake-use-git-branch-and-commit-details-in-project/

# get the current working branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# get the latest commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file(
    ${CMAKE_SOURCE_DIR}/version.hpp.in
    ${CMAKE_BINARY_DIR}/generated/version.hpp
)

include_directories(${CMAKE_BINARY_DIR}/generated)