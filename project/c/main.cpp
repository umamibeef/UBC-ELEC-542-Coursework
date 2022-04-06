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
 
#include <iostream>
#include <Eigen/Dense>
#include <boost/format.hpp>

#include "kernel.h"

typedef enum name
{
    IDX_X = 0,
    IDX_Y = 1,
    IDX_Z = 2,
    IDX_NUM,
} coordinateIndex_e;

void coordinate_index_to_coordinates(int num_partitions, int coordinate_index, int * coordinate_array)
{
    // convert coordinate index to x,y,z coordinates
    int base_10_num = coordinate_index;
    for (int i = 0; i < IDX_NUM; i++)
    {
        coordinate_array[i] = base_10_num % num_partitions;
        base_10_num /= num_partitions;
    }
}

void generate_coordinates(int num_partitions, int limit, double step_size, Eigen::Matrix<double, 1, Eigen::Dynamic> (&row_vector)[IDX_NUM])
{
    // Resize
    for (int i = 0; i < IDX_NUM; i++)
    {
        row_vector[i].resize(1, num_partitions);
    }
    // populate coordinates
    for (int i = 0; i < num_partitions; i++)
    {
        row_vector[IDX_X](i) = (double)(-limit) + (double)(i*step_size);
        row_vector[IDX_Y](i) = (double)(-limit) + (double)(i*step_size);
        row_vector[IDX_Z](i) = (double)(-limit) + (double)(i*step_size);
    }
}

// Generate the 3D Laplacian matrix for the given number of partitions
template <typename Type>
void generate_lapacian_matrix(Eigen::MatrixBase<Type> &matrix)
{
    // Matrix is square
    int matrix_dim = matrix.cols();
    // Number of partitions is cube root of matrix_dim
    int num_partitions = std::cbrt(matrix_dim);
    int row_coordinates[IDX_NUM];
    int col_coordinates[IDX_NUM];

    for (int row_coordinate_index = 0; row_coordinate_index < matrix_dim; row_coordinate_index++)
    {
        coordinate_index_to_coordinates(num_partitions, row_coordinate_index, row_coordinates);
        for (int col_coordinate_index = 0; col_coordinate_index < matrix_dim; col_coordinate_index++)
        {
            coordinate_index_to_coordinates(num_partitions, col_coordinate_index, col_coordinates);

            // U(x,y,z)
            if (row_coordinate_index == col_coordinate_index)
            {
                matrix(row_coordinate_index, col_coordinate_index) = -6.0;
            }

            if ((row_coordinates[IDX_Y] == col_coordinates[IDX_Y]) && (row_coordinates[IDX_Z] == col_coordinates[IDX_Z]))
            {
                // U(x-1,y,z)
                if (row_coordinates[IDX_X] == col_coordinates[IDX_X] + 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
                // U(x+1,y,z)
                if (row_coordinates[IDX_X] == col_coordinates[IDX_X] - 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
            }

            if ((row_coordinates[IDX_X] == col_coordinates[IDX_X]) && (row_coordinates[IDX_Z] == col_coordinates[IDX_Z]))
            {
                // U(x,y-1,z)
                if (row_coordinates[IDX_Y] == col_coordinates[IDX_Y] + 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
                // U(x,y+1,z)
                if (row_coordinates[IDX_Y] == col_coordinates[IDX_Y] - 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
            }

            if ((row_coordinates[IDX_X] == col_coordinates[IDX_X]) && (row_coordinates[IDX_Y] == col_coordinates[IDX_Y]))
            {
                // U(x,y,z-1)
                if (row_coordinates[IDX_Z] == col_coordinates[IDX_Z] + 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
                // U(x,y,z+1)
                if (row_coordinates[IDX_Z] == col_coordinates[IDX_Z] - 1)
                {
                    matrix(row_coordinate_index, col_coordinate_index) = 1.0;
                }
            }
        }
    }
}

int main(int argc, char ** argv)
{
    // number of partitions and limits
    int num_partitions = 11;
    int matrix_dim = num_partitions*num_partitions*num_partitions;
    int limit = 4;
    double step_size = (double)(4<<1)/(double)(num_partitions - 1);

    std::cout << boost::format("num_partitions = %d\n") % num_partitions;
    std::cout << boost::format("matrix_dim = %d\n") % matrix_dim;
    std::cout << boost::format("limit = %d\n") % limit;
    std::cout << boost::format("step_size = %f\n") % step_size;

    // coordinates
    Eigen::Matrix<double, 1, Eigen::Dynamic> coords[IDX_NUM];
    // generate coordinates (will resize and populate)
    generate_coordinates(num_partitions, limit, step_size, coords);

    std::cout << coords[IDX_X] << std::endl;
    std::cout << coords[IDX_Y] << std::endl;
    std::cout << coords[IDX_Z] << std::endl;

    // matrix instantiations
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> laplacian_matrix(matrix_dim, matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> kinetic_matrix(matrix_dim, matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> attraction_matrix(matrix_dim, matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> repulsion_matrix(matrix_dim, matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> exchange_matrix(matrix_dim, matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> fock_matrix(matrix_dim, matrix_dim);
    // can also use Eigen::MatrixXf matrix(rows, cols)

    // generate the second order Laplacian matrix for 3D space
    generate_lapacian_matrix(laplacian_matrix);

    // Call CUDA example
    cuda_example();

    return 0;
}