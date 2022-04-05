cmake_minimum_required(VERSION 3.16)

# set the program name
project(ExactHartreeFockSim)

# need GSL and CUDA
find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)
find_package(CUDA REQUIRED)

# add the executable
add_executable(ExactHartreeFockSim main.c)

# add compile options to enable GSL library
add_compile_options(`gsl-config --cflags --libs`)