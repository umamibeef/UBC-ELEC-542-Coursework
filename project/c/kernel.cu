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

// C/C++ includes
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

// Boost includes
#include <boost/format.hpp>

// Program includes
#include "main.hpp"
#include "config.hpp"
#include "console.hpp"
#include "kernel.h"

using namespace boost;

int multi_processor_count;
int max_blocks_per_multiprocessor;
int max_threads_per_multiprocessor;

// Electron-electron Coulombic repulsion function
__device__
float cuda_repulsion_function(LutVals_t lut_vals, int linear_coordinates_1, int linear_coordinates_2)
{
    const float epsilon = EPSILON;

    float x1 = lut_vals.coordinate_value_array[IDX_X * lut_vals.matrix_dim + linear_coordinates_1];
    float y1 = lut_vals.coordinate_value_array[IDX_Y * lut_vals.matrix_dim + linear_coordinates_1];
    float z1 = lut_vals.coordinate_value_array[IDX_Z * lut_vals.matrix_dim + linear_coordinates_1];

    float x2 = lut_vals.coordinate_value_array[IDX_X * lut_vals.matrix_dim + linear_coordinates_2];
    float y2 = lut_vals.coordinate_value_array[IDX_Y * lut_vals.matrix_dim + linear_coordinates_2];
    float z2 = lut_vals.coordinate_value_array[IDX_Z * lut_vals.matrix_dim + linear_coordinates_2];

    float denominator = sqrtf((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrtf(TINY_NUMBER);
    }

    return (1.0/(denominator));
}

__device__
float cuda_repulsion_matrix_integrand_function(LutVals_t lut_vals, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_2] * orbital_values[linear_coords_2] * cuda_repulsion_function(lut_vals, linear_coords_1, linear_coords_2);
}

__device__
float cuda_exchange_matrix_integrand_function(LutVals_t lut_vals, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1] * orbital_values[linear_coords_2] * cuda_repulsion_function(lut_vals, linear_coords_1, linear_coords_2);
}

__global__
void cuda_generate_repulsion_matrix_kernel(LutVals_t lut_vals, float *orbital_values, float *repulsion_diagonal)
{
    int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int electron_one_coordinate_index = start_index; electron_one_coordinate_index < lut_vals.matrix_dim; electron_one_coordinate_index += stride)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < lut_vals.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_repulsion_matrix_integrand_function(lut_vals, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        repulsion_diagonal[electron_one_coordinate_index] = sum * lut_vals.step_size_cubed;
    }
}

__global__
void cuda_generate_exchange_matrix_kernel(LutVals_t lut_vals, float *orbital_values, float *exchange_diagonal)
{
    int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int electron_one_coordinate_index = start_index; electron_one_coordinate_index < lut_vals.matrix_dim; electron_one_coordinate_index += stride)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < lut_vals.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_exchange_matrix_integrand_function(lut_vals, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        exchange_diagonal[electron_one_coordinate_index] = sum * lut_vals.step_size_cubed;
    }
}

int cuda_get_device_info(void)
{
    int num_devices;

    console_print_spacer(0, CLIENT_CUDA);
    console_print(0, "CUDA Device information:", CLIENT_CUDA);
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        console_print(0, str(format("Device Number: %d\n") % i), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Device name: %s\n") % prop.name), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "CUDA Capability %d.%d\n") % prop.major % prop.minor), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Memory Clock Rate (kHz): %d\n") % prop.memoryClockRate), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Memory Bus Width (bits): %d\n") % prop.memoryBusWidth), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Peak Memory Bandwidth (GB/s): %f\n") % (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6)), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Streaming Multiprocessors Count: %d\n") % prop.multiProcessorCount), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Max Blocks per Streaming Multiprocessors: %d\n") % prop.maxBlocksPerMultiProcessor), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Max Threads per Streaming Multiprocessors: %d\n") % prop.maxThreadsPerMultiProcessor), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Max Dimensions of a Thread Block (x,y,z): (%d, %d, %d)\n") % prop.maxThreadsDim[0] % prop.maxThreadsDim[1] % prop.maxThreadsDim[2]), CLIENT_CUDA);
        console_print(0, str(format(TAB1 "Max Dimensions of a Grid Size (x,y,z): (%d, %d, %d)\n") % prop.maxGridSize[0] % prop.maxGridSize[1] % prop.maxGridSize[2]), CLIENT_CUDA);

        // TODO: handle the multi CUDA device case
        multi_processor_count = prop.multiProcessorCount;
        max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
        max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    }
    if (num_devices == 0)
    {
        console_print_warn(0, "No CUDA devices available!", CLIENT_CUDA);
    }
    console_print_spacer(0, CLIENT_CUDA);

    return num_devices;
}

int cuda_allocate_shared_memory(LutVals_t *lut_vals, float **orbital_values_data, float **repulsion_diagonal_data, float **exchange_diagonal_data)
{
    int rv = 0;
    cudaError_t error;

    console_print(0, "Allocating unified memory for CPU/GPU...", CLIENT_CUDA);

    int orbital_vector_size_bytes = lut_vals->matrix_dim * sizeof(float);
    int repulsion_exchange_matrices_size_bytes = lut_vals->matrix_dim * sizeof(float);
    int coordinate_luts_size_bytes = IDX_NUM * lut_vals->matrix_dim * sizeof(float);

    cudaMallocManaged(orbital_values_data, orbital_vector_size_bytes);
    cudaMallocManaged(repulsion_diagonal_data, repulsion_exchange_matrices_size_bytes);
    cudaMallocManaged(exchange_diagonal_data, repulsion_exchange_matrices_size_bytes);

    cudaMallocManaged(&(lut_vals->coordinate_value_array), coordinate_luts_size_bytes);
    cudaMallocManaged(&(lut_vals->coordinate_index_array), coordinate_luts_size_bytes);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CLIENT_CUDA);
        rv = 1;
    }
    else
    {
        console_print(2, str(format("Allocated %d bytes for shared orbital values vector") % orbital_vector_size_bytes), CLIENT_CUDA);
        console_print(2, str(format("Allocated %d bytes for shared repulsion matrix diagonal") % repulsion_exchange_matrices_size_bytes), CLIENT_CUDA);
        console_print(2, str(format("Allocated %d bytes for shared exchange matrix diagonal") % repulsion_exchange_matrices_size_bytes), CLIENT_CUDA);
        console_print(2, str(format("Allocated 3x %d bytes for coordinate LUTs") % coordinate_luts_size_bytes), CLIENT_CUDA);
    }
    console_print_spacer(0, CLIENT_CUDA);

    return rv;
}

int cuda_free_shared_memory(LutVals_t *lut_vals, float **orbital_values_data, float **repulsion_diagonal_data, float **exchange_diagonal_data)
{
    int rv = 0;
    cudaError_t error;

    console_print(0, "Freeing unified memory for CPU/GPU...", CLIENT_CUDA);

    cudaFree(*orbital_values_data);
    cudaFree(*repulsion_diagonal_data);
    cudaFree(*exchange_diagonal_data);
    cudaFree(lut_vals->coordinate_value_array);
    cudaFree(lut_vals->coordinate_index_array);

    // null the pointers
    (*orbital_values_data) = nullptr;
    (*repulsion_diagonal_data) = nullptr;
    (*exchange_diagonal_data) = nullptr;
    (lut_vals->coordinate_value_array) = nullptr;
    (lut_vals->coordinate_index_array) = nullptr;

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CLIENT_CUDA);
        rv = 1;
    }
    else
    {
        console_print(2, "Successfully freed shared data", CLIENT_CUDA);
    }

    return rv;
}

int cuda_numerical_integration(LutVals_t lut_vals, float *orbital_values, float *repulsion_matrix, float *exchange_matrix)
{
    int rv = 0;
    cudaError_t error;

    int num_blocks = multi_processor_count * max_blocks_per_multiprocessor;
    int blocks_size = multi_processor_count * max_threads_per_multiprocessor / num_blocks;

    console_print(0, "Computing repulsion matrix", CLIENT_CUDA);
    cuda_generate_repulsion_matrix_kernel<<<num_blocks, blocks_size>>>(lut_vals, orbital_values, repulsion_matrix);
    cudaDeviceSynchronize();

    console_print(0, "Computing exchange matrix", CLIENT_CUDA);
    cuda_generate_exchange_matrix_kernel<<<num_blocks, blocks_size>>>(lut_vals, orbital_values, exchange_matrix);
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CLIENT_CUDA);
        rv = 1;
    }

    return rv;
}