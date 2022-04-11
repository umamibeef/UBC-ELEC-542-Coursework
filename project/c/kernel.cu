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

// Boost includes
#include <boost/format.hpp>

// Program includes
#include "main.hpp"
#include "config.hpp"
#include "console.hpp"
#include "kernel.h"

using namespace boost;

// Pointers to shared memory that we want to keep during program execution
float *orbital_values = nullptr;
float *repulsion_matrix = nullptr;
float *exchange_matrix = nullptr;

// TODO: The LUT might be slow since it's stored on the host, bring it to
// unified memory. In fact, we may need to bring the entire config struct into
// unified memory.

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
void cuda_generate_repulsion_matrix(LutVals_t lut_vals, float *orbital_values, float *repulsion_matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < lut_vals.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < lut_vals.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_repulsion_matrix_integrand_function(lut_vals, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        repulsion_matrix[electron_one_coordinate_index + electron_one_coordinate_index * lut_vals.matrix_dim] = sum * lut_vals.step_size_cubed;
    }
}

__global__
void cuda_generate_exchange_matrix(LutVals_t lut_vals, float *orbital_values, float *exchange_matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < lut_vals.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < lut_vals.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_exchange_matrix_integrand_function(lut_vals, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        exchange_matrix[electron_one_coordinate_index + electron_one_coordinate_index * lut_vals.matrix_dim] = sum * lut_vals.step_size_cubed;
    }
}

void cuda_print_device_info(void)
{
    int num_devices;

    console_print(0, "CUDA Device information:", CUDA);
    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        console_print(0, str(format("Device Number: %d\n") % i), CUDA);
        console_print(0, str(format(TAB1 "Device name: %s\n") % prop.name), CUDA);
        console_print(0, str(format(TAB1 "CUDA Capability %d.%d\n") % prop.major % prop.minor), CUDA);
        console_print(0, str(format(TAB1 "Memory Clock Rate (kHz): %d\n") % prop.memoryClockRate), CUDA);
        console_print(0, str(format(TAB1 "Memory Bus Width (bits): %d\n") % prop.memoryBusWidth), CUDA);
        console_print(0, str(format(TAB1 "Peak Memory Bandwidth (GB/s): %f\n") % (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6)), CUDA);
        console_print(0, str(format(TAB1 "MP Count: %d\n") % prop.multiProcessorCount), CUDA);
        console_print(0, str(format(TAB1 "Max Blocks per MP: %d\n") % prop.maxBlocksPerMultiProcessor), CUDA);
        console_print(0, str(format(TAB1 "Max Threads per MP: %d\n") % prop.maxThreadsPerMultiProcessor), CUDA);
        console_print(0, str(format(TAB1 "Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n") % prop.maxThreadsDim[0] % prop.maxThreadsDim[1] % prop.maxThreadsDim[2]), CUDA);
        console_print(0, str(format(TAB1 "Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n") % prop.maxGridSize[0] % prop.maxGridSize[1] % prop.maxGridSize[2]), CUDA);
    }
    console_print_spacer(0, CUDA);
}

int cuda_allocate_shared_memory(LutVals_t *lut_vals, float **orbital_values_data, float **repulsion_matrix_data, float **exchange_matrix_data)
{
    int rv = 0;
    cudaError_t error;

    console_print(0, "Allocating unified memory for CPU/GPU...", CUDA);

    int orbital_vector_size_bytes = lut_vals->matrix_dim * sizeof(float);
    int repulsion_exchange_matrices_size_bytes = lut_vals->matrix_dim * lut_vals->matrix_dim * sizeof(float);
    int coordinate_luts_size_bytes = IDX_NUM * lut_vals->matrix_dim * sizeof(float);

    cudaMallocManaged(orbital_values_data, orbital_vector_size_bytes);
    cudaMallocManaged(repulsion_matrix_data, repulsion_exchange_matrices_size_bytes);
    cudaMallocManaged(exchange_matrix_data, repulsion_exchange_matrices_size_bytes);

    cudaMallocManaged(&(lut_vals->coordinate_value_array), coordinate_luts_size_bytes);
    cudaMallocManaged(&(lut_vals->coordinate_index_array), coordinate_luts_size_bytes);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CUDA);
        rv = 1;
    }
    else
    {
        console_print(2, str(format("Allocated %d bytes for shared orbital values vector") % orbital_vector_size_bytes), CUDA);
        console_print(2, str(format("Allocated %d bytes for shared repulsion matrix") % repulsion_exchange_matrices_size_bytes), CUDA);
        console_print(2, str(format("Allocated %d bytes for shared exchange matrix") % repulsion_exchange_matrices_size_bytes), CUDA);
        console_print(2, str(format("Allocated 3x %d bytes for coordinate LUTs") % coordinate_luts_size_bytes), CUDA);
        console_print_spacer(2, CUDA);
    }

    return rv;
}

int cuda_free_shared_memory(LutVals_t *lut_vals, float **orbital_values_data, float **repulsion_matrix_data, float **exchange_matrix_data)
{
    int rv = 0;
    cudaError_t error;

    console_print(0, "Freeing unified memory for CPU/GPU...", CUDA);

    cudaFree(*orbital_values_data);
    cudaFree(*repulsion_matrix_data);
    cudaFree(*exchange_matrix_data);
    cudaFree(lut_vals->coordinate_value_array);
    cudaFree(lut_vals->coordinate_index_array);

    // null the pointers
    (*orbital_values_data) = nullptr;
    (*repulsion_matrix_data) = nullptr;
    (*exchange_matrix_data) = nullptr;
    (lut_vals->coordinate_value_array) = nullptr;
    (lut_vals->coordinate_index_array) = nullptr;

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CUDA);
        rv = 1;
    }
    else
    {
        console_print(2, "Successfully freed shared data", CUDA);
    }

    return rv;
}

int cuda_numerical_integration_kernel(LutVals_t lut_vals, float *orbital_values, float *repulsion_matrix, float *exchange_matrix)
{
    int rv = 0;
    cudaError_t error;

    if (!rv)
    {
        console_print(0, "Generating repulsion and exchange matrices on GPU...", CUDA);
        cuda_generate_repulsion_matrix<<<1,1>>>(lut_vals, orbital_values, repulsion_matrix);
        cuda_generate_exchange_matrix<<<1,1>>>(lut_vals, orbital_values, exchange_matrix);
        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();
    }

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        console_print_err(0, str(format("%s\n") % cudaGetErrorString(error)), CUDA);
        rv = 1;
    }

    return rv;
}