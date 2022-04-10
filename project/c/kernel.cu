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

// Electron-electron Coulombic repulsion function
float cuda_repulsion_function(cfg_t &config, int linear_coordinates_1, int linear_coordinates_2)
{
    const float epsilon = EPSILON;

    float x1 = config.coordinate_value_array[IDX_X][linear_coordinates_1];
    float y1 = config.coordinate_value_array[IDX_Y][linear_coordinates_1];
    float z1 = config.coordinate_value_array[IDX_Z][linear_coordinates_1];

    float x2 = config.coordinate_value_array[IDX_X][linear_coordinates_2];
    float y2 = config.coordinate_value_array[IDX_Y][linear_coordinates_2];
    float z2 = config.coordinate_value_array[IDX_Z][linear_coordinates_2];

    float denominator = sqrtf((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrtf(TINY_NUMBER);
    }

    return (1.0/(denominator));
}

float cuda_repulsion_matrix_integrand_function(cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_2]*orbital_values[linear_coords_2]*cuda_repulsion_function(config, linear_coords_1, linear_coords_2);
}

float cuda_exchange_matrix_integrand_function(cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1]*orbital_values[linear_coords_2]*cuda_repulsion_function(config, linear_coords_1, linear_coords_2);
}

void cuda_generate_repulsion_matrix(cfg_t &config, float *orbital_values, float *matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_repulsion_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*config.step_size_cubed;
    }
}


void cuda_generate_exchange_matrix(cfg_t &config, float *orbital_values, float *matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_exchange_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*config.step_size_cubed;
    }
}

void cuda_print_device_info(void)
{
    int num_devices;

    cudaGetDeviceCount(&num_devices);
    for (int i = 0; i < num_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        console_print(0, str(format(TAB1 "Device Number: %d\n") % i), CUDA);
        console_print(0, str(format(TAB2 "Device name: %s\n") % prop.name), CUDA);
        console_print(0, str(format(TAB2 "CUDA Capability %d.%d\n") % prop.major % prop.minor), CUDA);
        console_print(0, str(format(TAB2 "Memory Clock Rate (kHz): %d\n") % prop.memoryClockRate), CUDA);
        console_print(0, str(format(TAB2 "Memory Bus Width (bits): %d\n") % prop.memoryBusWidth), CUDA);
        console_print(0, str(format(TAB2 "Peak Memory Bandwidth (GB/s): %f\n") % (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6)), CUDA);
        console_print(0, str(format(TAB2 "MP Count: %d\n") % prop.multiProcessorCount), CUDA);
        console_print(0, str(format(TAB2 "Max Blocks per MP: %d\n") % prop.maxBlocksPerMultiProcessor), CUDA);
        console_print(0, str(format(TAB2 "Max Threads per MP: %d\n") % prop.maxThreadsPerMultiProcessor), CUDA);
        console_print(0, str(format(TAB2 "Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n") % prop.maxThreadsDim[0] % prop.maxThreadsDim[1] % prop.maxThreadsDim[2]), CUDA);
        console_print(0, str(format(TAB2 "Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n") % prop.maxGridSize[0] % prop.maxGridSize[1] % prop.maxGridSize[2]), CUDA);
    }
}

int cuda_numerical_integration_kernel(float * orbital_values)
{
    int rv = 0;
    cudaError_t error;

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("0 %s\n",cudaGetErrorString(error));
        rv = 1;
    }

    return rv;
}