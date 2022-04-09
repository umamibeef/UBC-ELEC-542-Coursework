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
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <locale>

// Eigen includes
#define EIGEN_USE_BLAS // use external BLAS routines
#define EIGEN_USE_LAPACKE_STRICT // use LAPACKE C interface to use LAPACK routines (STRICT: disable
                                 // less numerically robust routines)
#include <Eigen/Dense>
// This is a workaround needed to enable the use of EIGEN_USE_LAPACKE
// https://github.com/libigl/libigl/issues/651 for more info
#undef I

// Boost includes
#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

// OMP includes
#define OMP_NUM_THREADS (16)
#include <omp.h>

// CUDA
#include <cuda.h>

// Program includes
#include "main.hpp"
#include "kernel.h"

// Clean up code a bit by using aliases and typedefs
using namespace std;
using namespace boost;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_float_matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> eigen_float_col_vector;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> eigen_float_row_vector;

void console_print(int verbose_level, string input_string)
{
    time_t time_now = time(nullptr);
    char time_string[100];
    strftime(time_string, sizeof(time_string), "%Y/%m/%d-%H:%M:%S", localtime(&time_now));

    if (verbose_level > PROGRAM_VERBOSITY)
    {
        return;
    }

    stringstream ss(input_string);
    string output_string;
    while (getline(ss, output_string, '\n'))
    {
        cout << format("[%s]") % time_string << " " << output_string << endl;
    }
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

void linear_coordinate_index_to_spatial_coordinates_values(grid_cfg_t grid, int coordinate_index, float * coordinate_value_array)
{
    int coordinates_indices[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_index(grid, coordinate_index, coordinates_indices);
    coordinate_value_array[IDX_X] = (float)(-grid.limit) + ((float)coordinates_indices[IDX_X]*grid.step_size);
    coordinate_value_array[IDX_Y] = (float)(-grid.limit) + ((float)coordinates_indices[IDX_Y]*grid.step_size);
    coordinate_value_array[IDX_Z] = (float)(-grid.limit) + ((float)coordinates_indices[IDX_Z]*grid.step_size);
}

// Generate the 3D Laplacian matrix for the given number of partitions
template <typename Type>
void generate_laplacian_matrix(grid_cfg_t grid, Eigen::MatrixBase<Type> &matrix)
{
    int row_coordinates[IDX_NUM];
    int col_coordinates[IDX_NUM];

    matrix.setZero();

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
float attraction_function_helium(grid_cfg_t grid, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float coordinate_values[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates, coordinate_values);

    float x = coordinate_values[IDX_X];
    float y = coordinate_values[IDX_Y];
    float z = coordinate_values[IDX_Z];

    float denominator = sqrt(pow(x, 2.0) + pow(y + Y_OFFSET, 2.0) + pow(z + Z_OFFSET, 2.0));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (2.0/(denominator));
}

// Hydrogen molecule nucleus-electron Coulombic attraction function
float attraction_function_hydrogen(grid_cfg_t grid, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float coordinate_values[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates, coordinate_values);

    float x = coordinate_values[IDX_X];
    float y = coordinate_values[IDX_Y];
    float z = coordinate_values[IDX_Z];

    float denominator_1 = sqrt(pow((H2_BOND_LENGTH_ATOMIC_UNITS/2.0) - x, 2.0) + pow(y + Y_OFFSET, 2.0) + pow(z + Z_OFFSET, 2.0));
    float denominator_2 = sqrt(pow((-H2_BOND_LENGTH_ATOMIC_UNITS/2.0) - x, 2.0) + pow(y + Y_OFFSET, 2.0) + pow(z + Z_OFFSET, 2.0));

    if (abs(denominator_1) < epsilon)
    {
        denominator_1 = sqrt(TINY_NUMBER);
    }

    if (abs(denominator_2) < epsilon)
    {
        denominator_2 = sqrt(TINY_NUMBER);
    }

    return ((1.0/(denominator_1)) + (1.0/(denominator_2)));
}

// Generate the attraction matrix
template <typename Type>
void generate_attraction_matrix(grid_cfg_t grid, atomic_structure_e atomic_structure, Eigen::MatrixBase<Type> &matrix)
{
    matrix.setZero();

    // Create the diagonal vector
    eigen_float_col_vector attraction_matrix_diagonal(grid.matrix_dim, 1);

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

// Electron-electron Coulombic repulsion function
float repulsion_function(grid_cfg_t grid, int linear_coordinates_1, int linear_coordinates_2)
{
    const float epsilon = EPSILON;

    float coordinate_values_1[IDX_NUM];
    float coordinate_values_2[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates_1, coordinate_values_1);
    linear_coordinate_index_to_spatial_coordinates_values(grid, linear_coordinates_2, coordinate_values_2);

    float x1 = coordinate_values_1[IDX_X];
    float y1 = coordinate_values_1[IDX_Y];
    float z1 = coordinate_values_1[IDX_Z];

    float x2 = coordinate_values_2[IDX_X];
    float y2 = coordinate_values_2[IDX_Y];
    float z2 = coordinate_values_2[IDX_Z];

    float denominator = sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0) + pow(z2 - z1, 2.0));

    if (abs(denominator) < 1e-9)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (1.0/(denominator));
}

float repulsion_matrix_integrand_function(grid_cfg_t grid, float * orbital_values, int linear_coords_1, int linear_coords_2)
{
    return pow(orbital_values[linear_coords_2], 2.0)*repulsion_function(grid, linear_coords_1, linear_coords_2);
}

float exchange_matrix_integrand_function(grid_cfg_t grid, float * orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1]*orbital_values[linear_coords_2]*repulsion_function(grid, linear_coords_1, linear_coords_2);
}

void generate_repulsion_matrix(grid_cfg_t grid, float * orbital_values, float * matrix)
{
    float h_cubed = pow(grid.step_size, 3.0);

    // Copying what happens in the exchange matrix...
#if 0
    eigen_float_col_vector repulsion_matrix_diagonal(grid.matrix_dim, 1);
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < grid.matrix_dim; electron_two_coordinate_index++)
        {
            sum += repulsion_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        repulsion_matrix_diagonal(electron_one_coordinate_index) = sum*h_cubed;
    }
    matrix = repulsion_matrix_diagonal.asDiagonal();
#endif

    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < (electron_one_coordinate_index + 1); electron_two_coordinate_index++)
        {
            matrix[electron_one_coordinate_index + electron_two_coordinate_index*grid.matrix_dim] = repulsion_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index) * h_cubed;
            matrix[electron_two_coordinate_index + electron_one_coordinate_index*grid.matrix_dim] = matrix[electron_one_coordinate_index + electron_two_coordinate_index*grid.matrix_dim];
        }
    }
}


void generate_exchange_matrix(grid_cfg_t grid, float * orbital_values, float * matrix)
{
    float h_cubed = pow(grid.step_size, 3.0);

    // Why doesn't this work?! It's in the math!
#if 0
    eigen_float_col_vector exchange_matrix_diagonal(grid.matrix_dim, 1);
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < grid.matrix_dim; electron_two_coordinate_index++)
        {
            sum += exchange_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        exchange_matrix_diagonal(electron_one_coordinate_index) = sum*h_cubed;
    }
    matrix = exchange_matrix_diagonal.asDiagonal();
#endif
    
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < grid.matrix_dim; electron_one_coordinate_index++)
    {
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < (electron_one_coordinate_index + 1); electron_two_coordinate_index++)
        {
            // data in matrix is column major order, make sure we write the same way
            // e.g.:
            //  matrix in memory: 0 1 2 3 4 5 6 7 8
            //  data as matrix:
            //  0 3 6
            //  1 4 7  matrix(2,1) = 5
            //  2 5 8
            // matrix(electron_one_coordinate_index, electron_two_coordinate_index)
            // matrix(electron_two_coordinate_index, electron_one_coordinate_index)
            matrix[electron_one_coordinate_index + electron_two_coordinate_index*grid.matrix_dim] = exchange_matrix_integrand_function(grid, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index) * h_cubed;
            matrix[electron_two_coordinate_index + electron_one_coordinate_index*grid.matrix_dim] = matrix[electron_one_coordinate_index + electron_two_coordinate_index*grid.matrix_dim];
        }
    }
}

template <typename A, typename B, typename C, typename D, typename E>
float calculate_total_energy(Eigen::MatrixBase<A> &orbital_values, Eigen::MatrixBase<B> &kinetic_matrix, Eigen::MatrixBase<C> &attraction_matrix, Eigen::MatrixBase<D> &repulsion_matrix, Eigen::MatrixBase<E> &exchange_matrix)
{ 
    // orbital values are real, so no need to take conjugate
    eigen_float_row_vector psi_prime = orbital_values.transpose();
    eigen_float_col_vector psi = orbital_values;

    float energy_sum = 0;

    // sum for the total number of electrons in the systems: 2
    for (int i = 0; i < 2; i++)
    {
        auto hermitian_term = psi_prime*(-kinetic_matrix - attraction_matrix)*psi;
        auto hf_term = 0.5*psi_prime*(2*repulsion_matrix - exchange_matrix)*psi;
        energy_sum += hermitian_term(0) + hf_term(0);
    }

    return energy_sum;
}

// Using LAPACKE C wrapper for faster solving of eigenvalues and eigenvectors. A
// tutorial on how to do this is implemented here:
// https://eigen.tuxfamily.org/index.php?title=Lapack#Create_the_C.2B.2B_Function_Declaration.
// 
template <typename A, typename B>
bool lapack_solve_eigh_old(Eigen::Matrix<A, Eigen::Dynamic, Eigen::Dynamic> &matrix, Eigen::Matrix<B, Eigen::Dynamic, 1> &eigenvalues)
{
    int matrix_layout = LAPACK_COL_MAJOR; // column major ordering, default for eigen
    char jobz = 'V'; // compute eigenvalues and eigenvectors.
    char uplo = 'U'; // perform calculation on upper triangle of matrix
    lapack_int n = matrix.cols(); // order of the matrix (size)
    lapack_int lda = matrix.outerStride(); // the leading dimension of the array A. LDA >= max(1,N).
    lapack_int info = 0;

    console_print(2, str(format("\tn = matrix.cols() = %d") % n));
    console_print(2, str(format("\tlda = matrix.outerStride() = %d") % lda));

    float* a = matrix.data(); // pointer to fock/eigenvector data
    float* w = eigenvalues.data(); // pointer to eigenvalue data

    // float
    // lapack_int LAPACKE_ssyevd(int matrix_layout, char jobz, char uplo, lapack_int n,
    //                         float* a, lapack_int lda, float* w);
    // double
    // lapack_int LAPACKE_dsyevd(int matrix_layout, char jobz, char uplo, lapack_int n,
    //                         double* a, lapack_int lda, double* w);

    // Unfortunately this wrapper doesn't work for matrix sizes larger than a
    // certain amount,  because the querying reports back an incorrect value for
    // the work area.
    info = LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w);

    console_print(2, str(format("\tinfo = %d") % info));

    return (info == 0);
}

// Using the LAPACK routines directly by reimplementing the LAPACKE wrapper (to avoid the broken querying)
template <typename A, typename B>
bool lapack_solve_eigh(Eigen::Matrix<A, Eigen::Dynamic, Eigen::Dynamic> &matrix, Eigen::Matrix<B, Eigen::Dynamic, 1> &eigenvalues)
{
    int matrix_layout = LAPACK_COL_MAJOR; // column major ordering, default for eigen
    char jobz = 'V'; // compute eigenvalues and eigenvectors.
    char uplo = 'U'; // perform calculation on upper triangle of matrix
    lapack_int n = matrix.cols(); // order of the matrix (size)
    lapack_int lda = matrix.outerStride(); // the leading dimension of the array A. LDA >= max(1,N).
    lapack_int info = 0;
    lapack_int liwork;
    lapack_int lwork;
    lapack_int* iwork = NULL;
    float* work = NULL;
    float* a_ptr = matrix.data(); // pointer to fock/eigenvector data
    float* w_ptr = eigenvalues.data(); // pointer to eigenvalue data

    console_print(2, "\t** LAPACK solver debug");
    console_print(2, str(format("\tn = matrix.cols() = %d") % n));
    console_print(2, str(format("\tlda = matrix.outerStride() = %d") % lda));

    // Setting lwork and liwork manually based on the LAPACK documentation
    lwork = 1 + 6*n + 2*n*n;
    liwork = 3 + 5*n;

    console_print(2, str(format("\tliwork = %d") % liwork));
    console_print(2, str(format("\tlwork = %d") % lwork));

    // Allocate memory for work arrays
    iwork = (lapack_int*)LAPACKE_malloc(sizeof(lapack_int) * liwork);
    if(iwork == NULL)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print(2, "\tFATAL! Could not allocate iwork array");
    }
    work = (float*)LAPACKE_malloc(sizeof(float) * lwork);
    if(work == NULL)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print(2, "\tFATAL! Could not allocate work array");
    }

    // Call LAPACK function and adjust info if our work areas are OK
    if ((iwork != NULL) && (work != NULL))
    {
        console_print(2, "\tcalling LAPACK function");
        LAPACK_ssyevd(&jobz, &uplo, &n, a_ptr, &lda, w_ptr, work, &lwork, iwork, &liwork, &info);
        if( info < 0 ) {
            info = info - 1;
        }
    }

    // Release memory and exit
    LAPACKE_free(iwork);
    LAPACKE_free(work);

    console_print(2, str(format("\tinfo = %d") % info));
    return (info==0);
}

int main(int argc, char *argv[])
{
    console_print(0, "** Exact Hartree-Fock simulator **");

    omp_set_num_threads(OMP_NUM_THREADS); // Set the number of maximum threads to use for OMP
    Eigen::setNbThreads(OMP_NUM_THREADS); // Set the number of maximum threads to use for Eigen
    stringstream ss; // Used to form strings using cout

    // Program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("iterations", po::value<int>()->default_value(50), "set maximum number of iterations")
        ("partitions", po::value<int>()->default_value(11), "set number of partitions to divide solution space")
        ("limit", po::value<int>()->default_value(4), "set the solution space maximum x=y=z limit")
        ("convergence", po::value<float>()->default_value(0.1), "set the convergence condition (%)")
        ("structure", po::value<int>()->default_value(0), "set the atomic structure: (0:He, 1:H2)")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help"))
    {
        cout << desc << "\n";
        return 1;
    }

    atomic_structure_e atomic_structure;
    if (vm["structure"].as<int>() >= ATOMIC_STRUCTURE_NUM)
    {
        cout << "Invalid atomic structure selection: " << vm["structure"].as<int>() << endl;
        cout << "Valid options: (0:He, 1:H2)" << endl;
        return 0;
    }
    else
    {
        atomic_structure = (atomic_structure_e)(vm["structure"].as<int>());
    }

    // Program settings
    grid_cfg_t grid;
    grid.num_partitions = vm["partitions"].as<int>();
    grid.limit = vm["limit"].as<int>();
    grid.matrix_dim = grid.num_partitions*grid.num_partitions*grid.num_partitions;
    grid.step_size = (float)(4<<1)/(float)(grid.num_partitions - 1);
    int max_iterations = vm["iterations"].as<int>();
    int num_solutions = 6;
    float convergence_percentage = vm["convergence"].as<float>();

    console_print(0, str(format("\titerations = %d") % max_iterations));
    console_print(0, str(format("\tnum_partitions = %d") % grid.num_partitions));
    console_print(0, str(format("\tmatrix_dim = %d") % grid.matrix_dim));
    console_print(0, str(format("\tlimit = %d") % grid.limit));
    console_print(0, str(format("\tstep_size = %f") % grid.step_size));
    if (atomic_structure == HELIUM_ATOM)
    {
        console_print(0, "\tatomic structure: Helium Atom");
    }
    else if (atomic_structure == HYDROGEN_MOLECULE)
    {
        console_print(0, "\tatomic structure: Hydrogen Molecule");
    }

    // Matrix instantiations
    eigen_float_matrix laplacian_matrix(grid.matrix_dim, grid.matrix_dim);
    eigen_float_matrix kinetic_matrix(grid.matrix_dim, grid.matrix_dim);
    eigen_float_matrix attraction_matrix(grid.matrix_dim, grid.matrix_dim);
    eigen_float_matrix repulsion_matrix(grid.matrix_dim, grid.matrix_dim);
    eigen_float_matrix exchange_matrix(grid.matrix_dim, grid.matrix_dim);
    eigen_float_matrix fock_matrix(grid.matrix_dim, grid.matrix_dim);
    // Orbital values
    eigen_float_col_vector orbital_values(grid.matrix_dim, 1);
    // Eigenvectors and eigenvalues
    eigen_float_col_vector eigenvalues(grid.matrix_dim, 1);
    eigen_float_matrix eigenvectors(grid.matrix_dim, grid.matrix_dim);
    // Trimmed eigenvectors and eigenvalues
    eigen_float_col_vector trimmed_eigenvalues(num_solutions, 1);
    eigen_float_matrix trimmed_eigenvectors(grid.matrix_dim, num_solutions);

    console_print(0, "** Simulation start!");
    auto sim_start = chrono::system_clock::now();

    // generate the second order Laplacian matrix for 3D space
    console_print(1, "** Generating kinetic energy matrix");
    generate_laplacian_matrix(grid, laplacian_matrix);
    // generate the kinetic energy matrix
    kinetic_matrix = (laplacian_matrix/(2.0*grid.step_size*grid.step_size));
    // generate the Coulombic attraction matrix
    console_print(1, "** Generating electron-nucleus Coulombic attraction matrix");
    generate_attraction_matrix(grid, atomic_structure, attraction_matrix);

    // Main HF loop
    float last_total_energy = 0;
    float total_energy;
    float total_energy_percent_diff;
    int interation_count = 0;
    // initial solution
    fock_matrix = -kinetic_matrix - attraction_matrix;
    console_print(1, "** Obtaining eigenvalues and eigenvectors for initial solution...");
    // LAPACK solver
    if (!lapack_solve_eigh(fock_matrix, eigenvalues))
    {
        console_print(0, "** Something went horribly wrong with the solver, aborting");
        exit(EXIT_FAILURE);
    }

    orbital_values = eigenvectors.col(0);

    do
    {
        console_print(0, str(format("** Iteration: %d") % interation_count));

        auto iteration_start = chrono::system_clock::now();

        // generate repulsion matrix
        console_print(1, "** Generating electron-electron Coulombic repulsion matrix");
        generate_repulsion_matrix(grid, orbital_values.data(), repulsion_matrix.data());
        // generate exchange matrix
        console_print(1, "** Generating electron-electron exchange matrix");
        generate_exchange_matrix(grid, orbital_values.data(), exchange_matrix.data());
        // form fock matrix
        console_print(1, "** Generating Fock matrix");
        fock_matrix = -kinetic_matrix - attraction_matrix + 2.0*repulsion_matrix - exchange_matrix;

        console_print(1, "** Obtaining eigenvalues and eigenvectors...");

        // LAPACK solver, the same one used in numpy
        if (!lapack_solve_eigh(fock_matrix, eigenvalues))
        {
            console_print(0, "** Something went horribly wrong with the solver, aborting");
            exit(EXIT_FAILURE);
        }
        // using LAPACK solver, fock matrix now has the eigenvectors
        eigenvectors = fock_matrix;

        // Extract orbital_values
        orbital_values = eigenvectors.col(0);
        // Extract num_solutions eigenvalues
        trimmed_eigenvalues = eigenvalues.block(0, 0, num_solutions, 1);
        // Extract num_solutions eigenvectors
        trimmed_eigenvectors = eigenvectors.block(0, 0, grid.matrix_dim, num_solutions);

        total_energy = calculate_total_energy(orbital_values, kinetic_matrix, attraction_matrix, repulsion_matrix, exchange_matrix);
        total_energy_percent_diff = abs((total_energy - last_total_energy)/((total_energy + last_total_energy) / 2.0));

        console_print(0, str(format("** Total energy: %.3f") % (total_energy)));
        console_print(0, str(format("** Energy %% diff: %.3f%%") % (total_energy_percent_diff * 100.0)));

        // update last value
        last_total_energy = total_energy;

        // update iteration count
        interation_count++;

        auto iteration_end = chrono::system_clock::now();
        auto iteration_time = chrono::duration<float>(iteration_end - iteration_start);

        console_print(0, str(format("** Iteration end! Iteration time: %0.3f seconds**") % (float)(iteration_time.count())));

        // check if we've hit the maximum iteration limit
        if (interation_count == max_iterations)
        {
            break;
        }

        // check if we meet convergence condition
        if (abs(total_energy_percent_diff) < (convergence_percentage/100.0))
        {
            break;
        }

    } while(1);

    console_print(0, "** Final Eigenvalues:");
    ss << trimmed_eigenvalues;
    console_print(0, ss.str());
    ss.str(string()); // clear ss
    console_print(0, str(format("** Final Total energy: %.3f") % (total_energy)));

    auto sim_end = chrono::system_clock::now();
    auto sim_time = chrono::duration<float>(sim_end - sim_start);
    console_print(0, str(format("** Simulation end! Total time: %0.3f seconds**") % (float)(sim_time.count())));
    
    // Call CUDA example
    // cuda_example();

    return 0;
}