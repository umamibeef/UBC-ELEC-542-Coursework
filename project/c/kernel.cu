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

#include <stdlib.h>
#include <stdio.h>

#include "main.hpp"
#include "config.hpp"
#include "kernel.h"

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

    float denominator = sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0) + pow(z2 - z1, 2.0));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (1.0/(denominator));
}

float cuda_repulsion_matrix_integrand_function(cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return pow(orbital_values[linear_coords_2], 2.0)*cuda_repulsion_function(config, linear_coords_1, linear_coords_2);
}

float cuda_exchange_matrix_integrand_function(cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1]*orbital_values[linear_coords_2]*cuda_repulsion_function(config, linear_coords_1, linear_coords_2);
}

void cuda_generate_repulsion_matrix(cfg_t &config, float *orbital_values, float *matrix)
{
    float h_cubed = pow(config.step_size, 3.0);

    // Set matrix to 0
    memset(matrix,0.0,config.matrix_dim*config.matrix_dim*sizeof(float));

    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_repulsion_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*h_cubed;
    }
}


void cuda_generate_exchange_matrix(cfg_t &config, float *orbital_values, float *matrix)
{
    float h_cubed = pow(config.step_size, 3.0);

    // Set matrix to 0
    memset(matrix,0.0,config.matrix_dim*config.matrix_dim*sizeof(float));

    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += cuda_exchange_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*h_cubed;
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