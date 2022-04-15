/** 
MIT License

Copyright (c) [2022] [Michel Kakulphimp]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
**/

#pragma once

#include <vector>

// Atomic structure enumerator
typedef enum
{
    HELIUM_ATOM = 0,
    HYDROGEN_MOLECULE,
    ATOMIC_STRUCTURE_NUM,
} atomic_structure_e;

// Program configuration struct
typedef struct
{
    int verbosity; // The verbosity level of the program
    int num_partitions; // Number of partitions/quantizations for the grid
    int limit; // The x,y,z limits of the solution space
    int max_iterations; // Maximum number of HF interations to perform
    int num_solutions; // The number of eigenvalues and eigenvectors to keep from the solution
    float convergence_percentage; // The convergence percentage for total energy change, used for terminating the main loop
    atomic_structure_e atomic_structure; // The atomic structure to run the simulation against
    int max_num_threads; // Maximum number of threads Eigen can use for multiprocessing
    bool enable_cuda_integration; // Enable CUDA for the the numerical integration
    bool enable_cuda_eigensolver; // Enable CUDA for the eigensolver
    bool enable_csv_header_output; // Enable CSV output of simulation run for piping to file (disables other messages)
    bool enable_csv_data_all_output; // Enable CSV output (all data) of simulation run for piping to file (disables other messages)
    bool enable_csv_data_average_output; // Enable CSV output (average data) of simulation run for piping to file (disables other messages)
} Cfg_t;

// Lookup table struct for commonly used values
typedef struct
{
    size_t matrix_dim; // The resulting dimensions of the solution matrices
    float step_size; // The resulting step size for the given number of partitions and limits
    float step_size_cubed; // The step size cubed
    float *coordinate_value_array; // The linear coordinate to value LUT (array[matrix_dim][IDX_NUM])
    float *coordinate_index_array; // The linear coordinate to index LUT (array[matrix_dim][IDX_NUM])
} Lut_t;