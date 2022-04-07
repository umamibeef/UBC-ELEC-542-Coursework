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
 
// C++ includes
#include <iostream>
#include <cmath>
#include <chrono>

// Eigen includes
#define EIGEN_USE_LAPACKE // use LAPACKE C interface to use LAPACK routines
#include <Eigen/Dense>
// This is a workaround needed to enable the use of EIGEN_USE_LAPACKE
// https://github.com/libigl/libigl/issues/651 for more info
#undef I

// Boost includes
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time.hpp>

// Program includes
#include "main.hpp"
#include "kernel.h"

void console_print(std::string string)
{
    boost::posix_time::ptime time_now = boost::posix_time::second_clock::local_time();
    std::cout << boost::format("[%s]") % boost::posix_time::to_simple_string(time_now) << " " << string << std::endl;
}

void linear_coordinate_index_to_spatial_coordinates_index(grid_cfg_t grid, int coordinate_index, int * coordinate_index_array)
{
    // convert coordinate index to x,y,z coordinate indices
    int base_10_num = coordinate_index;
    for (int i = 0; i < IDX_NUM; i++)
    {
        coordinate_index_array[i] = base_10_num % grid.num_partitions;
        base_10_num /= grid.num_partitions;
    }
}

void linear_coordinate_index_to_spatial_coordinates_values(grid_cfg_t grid, int coordinate_index, double * coordinate_value_array)
{
    int coordinates_indices[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_index(grid, coordinate_index, coordinates_indices);
    coordinate_value_array[IDX_X] = (double)(-grid.limit) + ((double)coordinates_indices[IDX_X]*grid.step_size);
    coordinate_value_array[IDX_Y] = (double)(-grid.limit) + ((double)coordinates_indices[IDX_Y]*grid.step_size);
    coordinate_value_array[IDX_Z] = (double)(-grid.limit) + ((double)coordinates_indices[IDX_Z]*grid.step_size);
}

// TODO: Deprecate, no longer needed
template <typename Type>
void generate_coordinates(grid_cfg_t grid, Eigen::Matrix<Type, 1, Eigen::Dynamic> (&row_vector)[IDX_NUM])
{
    // Resize
    for (int i = 0; i < IDX_NUM; i++)
    {
        row_vector[i].resize(1, grid.num_partitions);
    }
    // populate coordinates
    for (int i = 0; i < grid.num_partitions; i++)
    {
        row_vector[IDX_X](i) = (double)(-grid.limit) + ((double)i*grid.step_size);
        row_vector[IDX_Y](i) = (double)(-grid.limit) + ((double)i*grid.step_size);
        row_vector[IDX_Z](i) = (double)(-grid.limit) + ((double)i*grid.step_size);
    }
}

// Generate the 3D Laplacian matrix for the given number of partitions
template <typename Type>
void generate_laplacian_matrix(grid_cfg_t grid, Eigen::MatrixBase<Type> &matrix)
{
    int row_coordinates[IDX_NUM];
    int col_coordinates[IDX_NUM];

    for (int row_coordinate_index = 0; row_coordinate_index < grid.matrix_dim; row_coordinate_index++)
    {
        linear_coordinate_index_to_spatial_coordinates_index(grid, row_coordinate_index, row_coordinates);
        for (int col_coordinate_index = 0; col_coordinate_index < grid.matrix_dim; col_coordinate_index++)
        {
            linear_coordinate_index_to_spatial_coordinates_index(grid, col_coordinate_index, col_coordinates);

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

// Helium atom nucleus-electron Coulombic attraction function
double attraction_function_helium(grid_cfg_t grid, int linear_coordinates)
{
    const double epsilon = EPSILON;

    double coordinate_values[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates, coordinate_values);

    double x = coordinate_values[IDX_X];
    double y = coordinate_values[IDX_Y];
    double z = coordinate_values[IDX_Z];

    double denominator = std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0) + std::pow(z + Z_OFFSET, 2.0));

    if (std::abs(denominator) < epsilon)
    {
        denominator = TINY_NUMBER;
    }

    return (2.0/(denominator));
}

// Hydrogen molecule nucleus-electron Coulombic attraction function
double attraction_function_hydrogen(grid_cfg_t grid, int linear_coordinates)
{
    const double epsilon = EPSILON;

    double coordinate_values[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates, coordinate_values);

    double x = coordinate_values[IDX_X];
    double y = coordinate_values[IDX_Y];
    double z = coordinate_values[IDX_Z];

    double denominator_1 = std::sqrt(std::pow((H2_BOND_LENGTH_ATOMIC_UNITS/2) - x, 2.0) + std::pow(y, 2.0) + std::pow(z + Z_OFFSET, 2.0));
    double denominator_2 = std::sqrt(std::pow((-H2_BOND_LENGTH_ATOMIC_UNITS/2) - x, 2.0) + std::pow(y, 2.0) + std::pow(z + Z_OFFSET, 2.0));

    if (std::abs(denominator_1) < epsilon)
    {
        denominator_1 = TINY_NUMBER;
    }

    if (std::abs(denominator_2) < epsilon)
    {
        denominator_2 = TINY_NUMBER;
    }

    return ((1.0/(denominator_1)) + (1.0/(denominator_2)));
}

// Electron-electron Coulombic repulsion function
double repulsion_function(grid_cfg_t grid, int linear_coordinates_1, int linear_coordinates_2)
{
    const double epsilon = EPSILON;

    double coordinate_values_1[IDX_NUM];
    double coordinate_values_2[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates_1, coordinate_values_1);
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates_2, coordinate_values_2);

    double x1 = coordinate_values_1[IDX_X];
    double y1 = coordinate_values_1[IDX_Y];
    double z1 = coordinate_values_1[IDX_Z];

    double x2 = coordinate_values_2[IDX_X];
    double y2 = coordinate_values_2[IDX_Y];
    double z2 = coordinate_values_2[IDX_Z];

    double denominator = std::sqrt(std::pow(x2 - x1, 2.0) + std::pow(y2 - y1, 2.0) + std::pow(z2 - z1, 2.0));

    if (std::abs(denominator) < epsilon)
    {
        denominator = TINY_NUMBER;
    }

    return (1.0/(denominator));
}

// Generate the attraction matrix
template <typename Type>
void generate_attraction_matrix(grid_cfg_t grid, atomic_structure_e atomic_structure, Eigen::MatrixBase<Type> &matrix)
{
    // Create the diagonal vector
    Eigen::Matrix<double, Eigen::Dynamic, 1> attraction_matrix_diagonal(grid.matrix_dim, 1);

    if (atomic_structure == HELIUM_ATOM)
    {
        for (int diagonal_index = 0; diagonal_index < grid.matrix_dim; diagonal_index++)
        {
            attraction_matrix_diagonal(diagonal_index) = attraction_function_helium(grid, diagonal_index);
        }
    }
    else if (atomic_structure == HYDROGEN_MOLECULE)
    {
        for (int diagonal_index = 0; diagonal_index < grid.matrix_dim; diagonal_index++)
        {
            attraction_matrix_diagonal(diagonal_index) = attraction_function_hydrogen(grid, diagonal_index);
        }
    }

    // copy the resulting diagonal matrix into the attraction matrix that was passed in
    matrix = attraction_matrix_diagonal.asDiagonal();
}

template <typename Type>
double repulsion_matrix_integrand_function(grid_cfg_t grid, const Eigen::MatrixBase<Type> &orbital_values, int linear_coords_1, int linear_coords_2)
{
    return std::pow(orbital_values(linear_coords_2), 2.0)*repulsion_function(grid, linear_coords_1, linear_coords_2);
}

template <typename Type>
double exchange_matrix_integrand_function(grid_cfg_t grid, const Eigen::MatrixBase<Type> &orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values(linear_coords_1)*orbital_values(linear_coords_2, 0)*repulsion_function(grid, linear_coords_1, linear_coords_2);
}

template <typename OrbitalType, typename MatrixType>
void generate_repulsion_matrix(grid_cfg_t grid, const Eigen::MatrixBase<OrbitalType> &orbital_values, Eigen::MatrixBase<MatrixType> &matrix)
{
    double sum = 0;
    Eigen::Matrix<double, Eigen::Dynamic, 1> repulsion_matrix_diagonal(grid.matrix_dim, 1);
    double h_cubed = std::pow(grid.step_size, 3.0);
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < grid.matrix_dim; electron_two_coordinate_index++)
        {
            sum += exchange_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index)*h_cubed;
        }
        repulsion_matrix_diagonal(electron_one_coordinate_index);
    }

    matrix = repulsion_matrix_diagonal.asDiagonal();
}

template <typename OrbitalType, typename MatrixType>
void generate_exchange_matrix(grid_cfg_t grid, const Eigen::MatrixBase<OrbitalType> &orbital_values, Eigen::MatrixBase<MatrixType> &matrix)
{
    double sum = 0;
    Eigen::Matrix<double, Eigen::Dynamic, 1> exchange_matrix_diagonal(grid.matrix_dim, 1);
    double h_cubed = std::pow(grid.step_size, 3.0);
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < grid.matrix_dim; electron_two_coordinate_index++)
        {
            sum += exchange_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index)*h_cubed;
        }
        exchange_matrix_diagonal(electron_one_coordinate_index);
    }

    matrix = exchange_matrix_diagonal.asDiagonal();
}

int main(int argc, char ** argv)
{
    console_print("** Exact Hartree-Fock simulator **");

    // number of partitions and limits
    grid_cfg_t grid;

    grid.num_partitions = 11;
    grid.matrix_dim = grid.num_partitions*grid.num_partitions*grid.num_partitions;
    grid.limit = 4;
    grid.step_size = (double)(4<<1)/(double)(grid.num_partitions - 1);

    console_print(boost::str(boost::format("\tnum_partitions = %d") % grid.num_partitions));
    console_print(boost::str(boost::format("\tmatrix_dim = %d") % grid.matrix_dim));
    console_print(boost::str(boost::format("\tlimit = %d") % grid.limit));
    console_print(boost::str(boost::format("\tstep_size = %f") % grid.step_size));

    // atomic structure to run simulation on
    // atomic_structure_e atomic_structure = HYDROGEN_MOLECULE;
    atomic_structure_e atomic_structure = HELIUM_ATOM;

    // matrix instantiations
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> laplacian_matrix(grid.matrix_dim, grid.matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> kinetic_matrix(grid.matrix_dim, grid.matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> attraction_matrix(grid.matrix_dim, grid.matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> repulsion_matrix(grid.matrix_dim, grid.matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> exchange_matrix(grid.matrix_dim, grid.matrix_dim);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> fock_matrix(grid.matrix_dim, grid.matrix_dim);
    
    // Orbital values (will initially be empty)
    Eigen::Matrix<double, Eigen::Dynamic, 1> orbital_values(grid.matrix_dim, 1);

    console_print("** Simulation start!");

    // generate the second order Laplacian matrix for 3D space
    console_print("** Generating kinetic energy matrix");
    generate_laplacian_matrix(grid, laplacian_matrix);
    // generate the kinetic energy matrix
    kinetic_matrix = (laplacian_matrix/(2.0*grid.step_size*grid.step_size));
    // generate the Coulombic attraction matrix
    console_print("** Generating electron-nucleus Coulombic attraction matrix");
    generate_attraction_matrix(grid, atomic_structure, attraction_matrix);

    // main HF loop
    int iteration_num = 0;
    do
    {
        console_print("** First iteration, zeros used as first guess");
        console_print(boost::str(boost::format("** Iteration: %d") % iteration_num));

        auto start = std::chrono::system_clock::now();

        // generate repulsion matrix
        console_print("** Generating electron-electron Coulombic repulsion matrix");
        generate_repulsion_matrix(grid, orbital_values, repulsion_matrix);
        // generate exchange matrix
        console_print("** Generating electron-electron exchange matrix");
        generate_exchange_matrix(grid, orbital_values, exchange_matrix);
        // form fock matrix
        console_print("** Generating Fock matrix");
        fock_matrix = -kinetic_matrix - attraction_matrix + 2.0*repulsion_matrix - exchange_matrix;

        console_print("** Obtaining eigenvalues and eigenvectors...");
        // Using the EigenSolver constructor to automatically compute() the eigenvalues and eigenvectors of fock_matrix
        Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> solver(fock_matrix);

        if (0)
        {
            // string stream for showning eigenvalues and eigenvectors
            std::stringstream ss;

            console_print("** Eigenvalues:");
            ss << solver.eigenvalues();
            console_print(ss.str());
            console_print("** Eigenvectors:");
            ss << solver.eigenvectors();
            console_print(ss.str());
        }

        auto end = std::chrono::system_clock::now();
        auto iteration_time = std::chrono::duration<double>(end - start);

        console_print(boost::str(boost::format("** Iteration end! Iteration time: %0.3f seconds**") % (double)(iteration_time.count())));

        break;
    } while(1);
    
    // Call CUDA example
    // cuda_example();

    return 0;
}