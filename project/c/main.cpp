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
#include <vector>

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
#include <omp.h>

// BLAS includes
#include <cblas.h>

// CUDA
#include <cuda.h>

// Program includes
#include "main.hpp"
#include "config.hpp"
#include "console.hpp"
#include "kernel.h"
#include "version.hpp"
#include "perfmon.hpp"

// Clean up code a bit by using aliases and typedefs
using namespace std;
using namespace boost;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> EigenFloatMatrix_t;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> EigenFloatColVector_t;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> EigenFloatRowVector_t;
typedef Eigen::Map<EigenFloatMatrix_t> EigenFloatMatrixMap_t;
typedef Eigen::Map<EigenFloatColVector_t> EigenFloatColVectorMap_t;

// Naughty naughty global!!!
int program_verbosity;

void parse_program_options(int argc, char **argv, Cfg_t &config)
{
    po::variables_map vm;
    po::options_description desc("EHFS options");
    stringstream ss;
    desc.add_options()
        ("help", "produce help message")
        ("iterations", po::value<int>()->default_value(20), "set maximum number of iterations")
        ("partitions", po::value<int>()->default_value(12), "set number of partitions to divide solution space")
        ("limit", po::value<int>()->default_value(4), "set the solution space maximum x=y=z limit")
        ("convergence", po::value<float>()->default_value(0.01), "set the convergence condition (%)")
        ("structure", po::value<int>()->default_value(0), "set the atomic structure: (0:He, 1:H2)")
        ("verbosity", po::value<int>()->default_value(2), "set the verbosity of the program")
        ("max-threads", po::value<int>()->default_value(16), "set the maximum number of threads that CPU operations can spawn")
        ("use-gpu-int", po::value<bool>()->default_value(1), "enable CUDA GPU acceleration for numerical integration")
        ("use-gpu-eig", po::value<bool>()->default_value(1), "enable CUDA GPU acceleration for the eigensolver")
        ("csv-header", po::value<bool>()->default_value(0), "enable CSV output (header) of simulation run for piping to file (disables other messages)")
        ("csv-data-all", po::value<bool>()->default_value(0), "enable CSV output (all data) of simulation run for piping to file (disables other messages)")
        ("csv-data-avg", po::value<bool>()->default_value(0), "enable CSV output (average data) of simulation run for piping to file (disables other messages)")
    ;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    if (vm.count("help"))
    {
        ss << desc << "\n";
        console_print(0, ss.str(), CLIENT_SIM);
        exit(1);
    }

    if (vm["structure"].as<int>() >= ATOMIC_STRUCTURE_NUM)
    {
        ss << "Invalid atomic structure selection: " << vm["structure"].as<int>() << endl;
        ss << "Valid options: (0:He, 1:H2)" << endl;
        console_print(0, ss.str(), CLIENT_SIM);
        exit(0);
    }
    else
    {
        config.atomic_structure = (atomic_structure_e)(vm["structure"].as<int>());
    }

    // Program configuration from CLI
    config.num_partitions = vm["partitions"].as<int>();
    config.limit = vm["limit"].as<int>();
    config.max_iterations = vm["iterations"].as<int>();
    config.num_solutions = 6;
    config.convergence_percentage = vm["convergence"].as<float>();
    config.verbosity = vm["verbosity"].as<int>();
    config.max_num_threads = vm["max-threads"].as<int>();
    config.enable_cuda_integration = vm["use-gpu-int"].as<bool>();
    config.enable_cuda_eigensolver = vm["use-gpu-eig"].as<bool>();
    config.enable_csv_header_output = vm["csv-header"].as<bool>();
    config.enable_csv_data_all_output = vm["csv-data-all"].as<bool>();
    config.enable_csv_data_average_output = vm["csv-data-avg"].as<bool>();
    if (config.enable_csv_header_output || config.enable_csv_data_all_output || config.enable_csv_data_average_output)
    {
        config.verbosity = -1;
    }
    program_verbosity = config.verbosity;
}

void print_header(void)
{
    stringstream ss;

    ss  << ANSI_FG_COLOR_MAGENTA "                                      " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   ███████╗██╗  ██╗███████╗███████╗   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   ██╔════╝██║  ██║██╔════╝██╔════╝   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   █████╗  ███████║█████╗  ███████╗   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   ██╔══╝  ██╔══██║██╔══╝  ╚════██║   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   ███████╗██║  ██║██║     ███████║   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "   ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝   " ANSI_COLOR_RESET << endl
        << ANSI_FG_COLOR_MAGENTA "                                      " ANSI_COLOR_RESET << endl
        <<                       "   < Exact Hartree-Fock Simulator >   " << endl << endl
        << "Author: Michel Kakulphimp" << endl
        << "Build Date: " __DATE__ " " __TIME__ << endl
        << "Git Branch: " << GIT_BRANCH << endl
        << "Git Hash: " << GIT_COMMIT_HASH << endl << endl;

    console_print_hr(0, CLIENT_SIM);
    console_print(0, ss.str(), CLIENT_SIM);
}

void populate_lookup_values(Cfg_t &config, Lut_t &lut)
{
    // Memory for the arrays filled out in this function should have been
    // allocated already. Maybe check for null later. TODO.

    // Populate LUTs
    for (size_t coordinate_index = 0; coordinate_index < lut.matrix_dim; coordinate_index++)
    {
        // convert coordinate index to x,y,z coordinate indices
        int base_10_num = coordinate_index;
        for (int i = 0; i < IDX_NUM; i++)
        {
            lut.coordinate_index_array[i * lut.matrix_dim + coordinate_index] =
                base_10_num % config.num_partitions;
            base_10_num /= config.num_partitions;

            lut.coordinate_value_array[i * lut.matrix_dim + coordinate_index] =
                (float)(-config.limit) + ((float)lut.coordinate_index_array[i * lut.matrix_dim + coordinate_index]
                    * lut.step_size);
        }
    }
}

// Generate the 3D Laplacian matrix for the given number of partitions
void generate_laplacian_matrix(Lut_t lut, float *matrix)
{
    size_t col_index_x;
    size_t col_index_y;
    size_t col_index_z;
    size_t row_index_x;
    size_t row_index_y;
    size_t row_index_z;

    for (size_t row_coordinate_index = 0; row_coordinate_index < lut.matrix_dim; row_coordinate_index++)
    {
        for (size_t col_coordinate_index = 0; col_coordinate_index < lut.matrix_dim; col_coordinate_index++)
        {
            col_index_x = lut.coordinate_index_array[IDX_X * lut.matrix_dim + col_coordinate_index];
            col_index_y = lut.coordinate_index_array[IDX_Y * lut.matrix_dim + col_coordinate_index];
            col_index_z = lut.coordinate_index_array[IDX_Z * lut.matrix_dim + col_coordinate_index];

            row_index_x = lut.coordinate_index_array[IDX_X * lut.matrix_dim + row_coordinate_index];
            row_index_y = lut.coordinate_index_array[IDX_Y * lut.matrix_dim + row_coordinate_index];
            row_index_z = lut.coordinate_index_array[IDX_Z * lut.matrix_dim + row_coordinate_index];

            // U(x,y,z)
            if (row_coordinate_index == col_coordinate_index)
            {
                matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = -6.0;
            }

            if ((row_index_y == col_index_y) && (row_index_z == col_index_z))
            {
                // U(x-1,y,z)
                if (row_index_x == col_index_x + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
                // U(x+1,y,z)
                if (row_index_x == col_index_x - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
            }

            if ((row_index_x == col_index_x) && (row_index_z == col_index_z))
            {
                // U(x,y-1,z)
                if (row_index_y == col_index_y + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
                // U(x,y+1,z)
                if (row_index_y == col_index_y - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
            }

            if ((row_index_x == col_index_x) && (row_index_y == col_index_y))
            {
                // U(x,y,z-1)
                if (row_index_z == col_index_z + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
                // U(x,y,z+1)
                if (row_index_z == col_index_z - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*lut.matrix_dim] = 1.0;
                }
            }
        }
    }
}

// Helium atom nucleus-electron Coulombic attraction function
float attraction_function_helium(Lut_t lut, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float x = lut.coordinate_value_array[IDX_X * lut.matrix_dim + linear_coordinates];
    float y = lut.coordinate_value_array[IDX_Y * lut.matrix_dim + linear_coordinates];
    float z = lut.coordinate_value_array[IDX_Z * lut.matrix_dim + linear_coordinates];

    float denominator = sqrt(pow(x, 2.0) + pow(y + Y_OFFSET, 2.0) + pow(z + Z_OFFSET, 2.0));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (2.0/(denominator));
}

// Hydrogen molecule nucleus-electron Coulombic attraction function
float attraction_function_hydrogen(Lut_t lut, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float x = lut.coordinate_value_array[IDX_X * lut.matrix_dim + linear_coordinates];
    float y = lut.coordinate_value_array[IDX_Y * lut.matrix_dim + linear_coordinates];
    float z = lut.coordinate_value_array[IDX_Z * lut.matrix_dim + linear_coordinates];

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
void generate_attraction_matrix(Lut_t lut, atomic_structure_e atomic_structure, float *matrix)
{
    if (atomic_structure == HELIUM_ATOM)
    {
        for (size_t diagonal_index = 0; diagonal_index < lut.matrix_dim; diagonal_index++)
        {
            matrix[diagonal_index + diagonal_index*lut.matrix_dim] = attraction_function_helium(lut, diagonal_index);
        }
    }
    else if (atomic_structure == HYDROGEN_MOLECULE)
    {
        for (size_t diagonal_index = 0; diagonal_index < lut.matrix_dim; diagonal_index++)
        {
            matrix[diagonal_index + diagonal_index*lut.matrix_dim] = attraction_function_hydrogen(lut, diagonal_index);
        }
    }
}

// Electron-electron Coulombic repulsion function
float repulsion_function(Lut_t lut, int linear_coordinates_1, int linear_coordinates_2)
{
    const float epsilon = EPSILON;

    float x1 = lut.coordinate_value_array[IDX_X * lut.matrix_dim + linear_coordinates_1];
    float y1 = lut.coordinate_value_array[IDX_Y * lut.matrix_dim + linear_coordinates_1];
    float z1 = lut.coordinate_value_array[IDX_Z * lut.matrix_dim + linear_coordinates_1];

    float x2 = lut.coordinate_value_array[IDX_X * lut.matrix_dim + linear_coordinates_2];
    float y2 = lut.coordinate_value_array[IDX_Y * lut.matrix_dim + linear_coordinates_2];
    float z2 = lut.coordinate_value_array[IDX_Z * lut.matrix_dim + linear_coordinates_2];

    float denominator = sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0) + pow(z2 - z1, 2.0));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (1.0/(denominator));
}

float repulsion_diagonal_integrand_function(Lut_t lut, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return pow(orbital_values[linear_coords_2], 2.0)*repulsion_function(lut, linear_coords_1, linear_coords_2);
}

float exchange_diagonal_integrand_function(Lut_t lut, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1]*orbital_values[linear_coords_2]*repulsion_function(lut, linear_coords_1, linear_coords_2);
}

void generate_repulsion_diagonal(Lut_t lut, DynamicDataPointers_t ddp)
{
    for (size_t electron_one_coordinate_index = 0; electron_one_coordinate_index < lut.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (size_t electron_two_coordinate_index = 0; electron_two_coordinate_index < lut.matrix_dim; electron_two_coordinate_index++)
        {
            sum += repulsion_diagonal_integrand_function(lut, ddp.orbital_values_data, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        ddp.repulsion_diagonal_data[electron_one_coordinate_index] = sum*lut.step_size_cubed;
    }
}


void generate_exchange_diagonal(Lut_t lut, DynamicDataPointers_t ddp)
{
    for (size_t electron_one_coordinate_index = 0; electron_one_coordinate_index < lut.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (size_t electron_two_coordinate_index = 0; electron_two_coordinate_index < lut.matrix_dim; electron_two_coordinate_index++)
        {
            sum += exchange_diagonal_integrand_function(lut, ddp.orbital_values_data, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        ddp.exchange_diagonal_data[electron_one_coordinate_index] = sum*lut.step_size_cubed;
    }
}

template <typename A, typename B, typename C, typename D, typename E>
float calculate_total_energy(Eigen::MatrixBase<A> &orbital_values, Eigen::MatrixBase<B> &kinetic_matrix, Eigen::MatrixBase<C> &attraction_matrix, Eigen::MatrixBase<D> &repulsion_diagonal, Eigen::MatrixBase<E> &exchange_diagonal)
{ 
    // orbital values are real, so no need to take conjugate
    EigenFloatRowVector_t psi_prime = orbital_values.transpose();
    EigenFloatColVector_t psi = orbital_values;

    EigenFloatMatrix_t repulsion_matrix = repulsion_diagonal.asDiagonal();
    EigenFloatMatrix_t exchange_matrix = exchange_diagonal.asDiagonal();

    float energy_sum = 0;

    console_print_hr(1, CLIENT_SIM);
    console_print(1, "Calculating total energy", CLIENT_SIM);

    // sum for the total number of electrons in the systems: 2
    for (int i = 0; i < 2; i++)
    {
        auto hermitian_term = psi_prime * (-kinetic_matrix - attraction_matrix) * psi;
        auto hf_term = 0.5 * psi_prime * (2.0 * repulsion_matrix - exchange_matrix) * psi;
        energy_sum += (hermitian_term(0) + hf_term(0));
    }

    return energy_sum;
}

// Using the LAPACK routines directly by reimplementing the LAPACKE wrapper
// without querying for sizes. I was originally using LAPACKE_ssyevd directly
// from LAPACKE.h, but for some reason, the automatic work size was returning an
// invalid number. I'm re-implementing the core of that function in this
// version, but I'm using manual lwork and liwork calculations instead. From the
// LAPACK documentation on ssyevd(): SSYEVD computes all eigenvalues and,
// optionally, eigenvectors of a real symmetric matrix A. If eigenvectors are
// desired, it uses a divide and conquer algorithm.The divide and conquer
// algorithm makes very mild assumptions about floating point arithmetic.
//
// @param[in]  lut     Program constants
// @param      matrix       The matrix originally containing the matrix to be
//                          solved and the resulting eigenvectors
// @param      eigenvalues  The eigenvalues of the solution
//
// @return     True if the solver found solutions, false if the solver failed
//             for some reason
//
bool lapack_solve_eigh(Lut_t &lut, DynamicDataPointers_t ddp)
{
    char jobz = 'V'; // compute eigenvalues and eigenvectors.
    char uplo = 'U'; // perform calculation on upper triangle of matrix
    lapack_int n = lut.matrix_dim; // order of the matrix (size)
    lapack_int lda = lut.matrix_dim; // the leading dimension of the array A. LDA >= max(1,N).
    lapack_int info = 0;
    lapack_int liwork;
    lapack_int lwork;
    lapack_int* iwork = nullptr;
    float* work = nullptr;

    auto lapack_start = chrono::system_clock::now();

    console_print(0, "LAPACK ssyevd start", CLIENT_LAPACK);
    console_print(2, TAB1 "LAPACK solver debug", CLIENT_LAPACK);
    console_print(2, str(format(TAB2 "n = %d") % n), CLIENT_LAPACK);
    console_print(2, str(format(TAB2 "lda = %d") % lda), CLIENT_LAPACK);

    // Setting lwork and liwork manually based on the LAPACK documentation
    lwork = 1 + 6*n + 2*n*n;
    liwork = 3 + 5*n;

    console_print(2, str(format(TAB2 "liwork = %d") % liwork), CLIENT_LAPACK);
    console_print(2, str(format(TAB2 "lwork = %d") % lwork), CLIENT_LAPACK);

    // Allocate memory for work arrays
    iwork = (lapack_int*)malloc(sizeof(lapack_int) * liwork);
    if(iwork == nullptr)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print_err(2, TAB2 "FATAL! Could not allocate iwork array", CLIENT_LAPACK);
    }
    work = (float*)malloc(sizeof(float) * lwork);
    if(work == nullptr)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print_err(2, TAB2 "FATAL! Could not allocate work array", CLIENT_LAPACK);
    }

    // Call LAPACK function and adjust info if our work areas are OK
    if ((iwork != nullptr) && (work != nullptr))
    {
        console_print(2, TAB2 "calling LAPACK function", CLIENT_LAPACK);
        LAPACK_ssyevd(&jobz, &uplo, &n, ddp.eigenvectors_data, &lda, ddp.eigenvalues_data, work, &lwork, iwork, &liwork, &info);
        if( info < 0 ) {
            info = info - 1;
        }
    }

    // Release memory and exit
    LAPACKE_free(iwork);
    LAPACKE_free(work);

    console_print(2, str(format(TAB2 "info = %d") % info), CLIENT_LAPACK);

    auto lapack_end = chrono::system_clock::now();
    auto lapack_time = chrono::duration<float>(lapack_end - lapack_start);

    console_print(0, str(format("LAPACK ssyevd took: %0.3f seconds") % (float)(lapack_time.count())), CLIENT_LAPACK);

    return (info==0);
}

int cpu_allocate_integration_memory(Lut_t &lut, DynamicDataPointers_t &ddp)
{
    int rv = 0;

    console_print_hr(2, CLIENT_SIM);
    console_print(2, "Allocating memory for CPU integration", CLIENT_SIM);

    size_t orbital_vector_size_bytes = lut.matrix_dim;
    size_t repulsion_exchange_matrices_size_bytes = lut.matrix_dim;
    size_t coordinate_luts_size_bytes = IDX_NUM * lut.matrix_dim;

    ddp.orbital_values_data = (float*)(calloc(orbital_vector_size_bytes, sizeof(float)));
    ddp.repulsion_diagonal_data = (float*)(calloc(repulsion_exchange_matrices_size_bytes, sizeof(float)));
    ddp.exchange_diagonal_data = (float*)(calloc(repulsion_exchange_matrices_size_bytes, sizeof(float)));

    lut.coordinate_value_array = (float*)(calloc(coordinate_luts_size_bytes, sizeof(float)));
    lut.coordinate_index_array = (float*)(calloc(coordinate_luts_size_bytes, sizeof(float)));

    if ( (ddp.orbital_values_data == nullptr) || (ddp.repulsion_diagonal_data == nullptr) || (ddp.exchange_diagonal_data == nullptr) ||
         (lut.coordinate_value_array == nullptr) || (lut.coordinate_index_array == nullptr) )
    {
        console_print_err(0, "Memory allocation error!", CLIENT_SIM);
        rv = 1;
    }
    else
    {
        console_print(2, str(format("Allocated %d bytes for orbital values vector") % orbital_vector_size_bytes), CLIENT_SIM);
        console_print(2, str(format("Allocated %d bytes for repulsion matrix diagonal") % repulsion_exchange_matrices_size_bytes), CLIENT_SIM);
        console_print(2, str(format("Allocated %d bytes for exchange matrix diagonal") % repulsion_exchange_matrices_size_bytes), CLIENT_SIM);
        console_print(2, str(format("Allocated 3x %d bytes for coordinate LUTs") % coordinate_luts_size_bytes), CLIENT_SIM);
    }

    return rv;
}

int cpu_allocate_eigensolver_memory(Lut_t &lut, DynamicDataPointers_t &ddp)
{
    int rv = 0;

    console_print_hr(2, CLIENT_SIM);
    console_print(2, "Allocating memory for CPU eigensolver", CLIENT_SIM);

    size_t eigenvectors_size_bytes = sizeof(float) * lut.matrix_dim * lut.matrix_dim;
    size_t eigenvalues_size_bytes = sizeof(float) * lut.matrix_dim;

    ddp.eigenvectors_data = (float*)(malloc(eigenvectors_size_bytes));
    ddp.eigenvalues_data = (float*)(malloc(eigenvalues_size_bytes));

    if ((ddp.eigenvectors_data == nullptr) || (ddp.eigenvalues_data == nullptr))
    {
        console_print_err(0, "Memory allocation error!", CLIENT_SIM);
        rv = 1;
    }
    else
    {
        console_print(2, str(format("Allocated %d bytes for eigenvector matrix") % eigenvectors_size_bytes), CLIENT_SIM);
        console_print(2, str(format("Allocated %d bytes for eigenvalue vector") % eigenvalues_size_bytes), CLIENT_SIM);
    }

    return rv;
}

int cpu_free_integration_memory(Lut_t &lut, DynamicDataPointers_t &ddp)
{
    int rv = 0;

    console_print(2, "Freeing allocated integration memory...", CLIENT_SIM);

    free(ddp.orbital_values_data);
    free(ddp.repulsion_diagonal_data);
    free(ddp.exchange_diagonal_data);
    free(lut.coordinate_value_array);
    free(lut.coordinate_index_array);

    // null the pointers
    (ddp.orbital_values_data) = nullptr;
    (ddp.repulsion_diagonal_data) = nullptr;
    (ddp.exchange_diagonal_data) = nullptr;
    (lut.coordinate_value_array) = nullptr;
    (lut.coordinate_index_array) = nullptr;

    console_print(2, "Successfully freed allocated integration memory", CLIENT_SIM);

    return rv;
}

int cpu_free_eigensolver_memory(DynamicDataPointers_t &ddp)
{
    int rv = 0;

    console_print(2, "Freeing allocated eigensolver memory...", CLIENT_SIM);

    free(ddp.eigenvectors_data);
    free(ddp.eigenvalues_data);

    // null the pointers
    ddp.eigenvectors_data = nullptr;
    ddp.eigenvalues_data = nullptr;

    console_print(2, "Successfully freed allocated eigensolver memory", CLIENT_SIM);

    return rv;
}

void print_program_configurations(Cfg_t &config, Lut_t &lut)
{
    // Print program information
    console_print_hr(0, CLIENT_SIM);
    console_print(0, "Program Configurations:\n", CLIENT_SIM);
    console_print(0, str(format(TAB1 "Verbosity = %d") % config.verbosity), CLIENT_SIM);
    console_print(0, str(format(TAB1 "Iterations = %d") % config.max_iterations), CLIENT_SIM);
    console_print(0, str(format(TAB1 "Num Partitions = %d") % config.num_partitions), CLIENT_SIM);
    console_print(0, str(format(TAB1 "Limits = %d") % config.limit), CLIENT_SIM);
    console_print(0, str(format(TAB1 "Matrix Dimension = %d") % lut.matrix_dim), CLIENT_SIM);
    console_print(0, str(format(TAB1 "Step Size = %f") % lut.step_size), CLIENT_SIM);

    if (config.atomic_structure == HELIUM_ATOM)
    {
        console_print(0, TAB1 "Atomic Structure: Helium Atom", CLIENT_SIM);
    }
    else if (config.atomic_structure == HYDROGEN_MOLECULE)
    {
        console_print(0, TAB1 "Atomic Structure: Hydrogen Molecule", CLIENT_SIM);
    }
    console_print(0, str(format(TAB1 "Maximum CPU Threads = %d") % config.max_num_threads), CLIENT_SIM);
}

void config_cuda(Cfg_t &config)
{
    if (config.enable_cuda_eigensolver || config.enable_cuda_integration)
    {
        console_print(0, "Checking for available CUDA devices", CLIENT_SIM);
        int num_cuda_devices = cuda_get_device_info();
        if ((num_cuda_devices == 0) && (config.enable_cuda_eigensolver || config.enable_cuda_integration))
        {
            console_print_warn(0, "Disabling CUDA acceleration since no device is available", CLIENT_SIM);
            config.enable_cuda_integration = false;
            config.enable_cuda_eigensolver = false;
        }
    }
}

// Solve for Fock matrix eigenvectors
void eigensolver(Cfg_t config, Lut_t &lut, PerformanceMonitor &perfmon, DynamicDataPointers_t ddp)
{
    console_print(1, "Obtaining eigenvalues and eigenvectors", CLIENT_SIM);

    auto eig_start = chrono::system_clock::now();

    if (config.enable_cuda_eigensolver)
    {
        // CUDA cuSOLVER
        if (!cuda_eigensolver(lut, ddp))
        {
            console_print_err(0, "Something went horribly wrong with the CUDA solver, aborting", CLIENT_SIM);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // Eigen solver, slow.
        // EigenFloatMatrixMap_t eigenvectors(eigenvectors_data, lut.matrix_dim, lut.matrix_dim);
        // EigenFloatColVectorMap_t eigenvalues(eigenvalues_data, lut.matrix_dim, 1);
        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> solver(eigenvectors);
        // eigenvectors = solver.eigenvectors();
        // eigenvalues = solver.eigenvalues();

        // CPU LAPACK solver
        if (!lapack_solve_eigh(lut, ddp))
        {
            console_print_err(0, "Something went horribly wrong with the LAPACK solver, aborting", CLIENT_SIM);
            exit(EXIT_FAILURE);
        }
    }

    auto eig_end = chrono::system_clock::now();
    auto eig_time = chrono::duration<float>(eig_end - eig_start);
    perfmon.record(PerformanceMonitor::ITERATION_EIGENSOLVER_TIME, (float)(eig_time.count()));
    console_print(0, str(format("Eigenvalues and eigenvectors computed in: %0.3f seconds") % (float)(eig_time.count())), CLIENT_SIM);
}

void generate_repulsion_and_integration_matrices(Cfg_t &config, Lut_t &lut, PerformanceMonitor &perfmon, DynamicDataPointers_t ddp)
{
    auto int_start = chrono::system_clock::now();

    if (config.enable_cuda_integration)
    {
        console_print(0, "Generating repulsion and exchange matrices on GPU", CLIENT_SIM);

        // generate repulsion and exchange matrices on GPU
        cuda_numerical_integration(lut, ddp);
    }
    else
    {
        console_print(0, "Generating repulsion and exchange matrices on CPU", CLIENT_SIM);

        // generate repulsion and exchange matrices on CPU
        generate_repulsion_diagonal(lut, ddp);
        generate_exchange_diagonal(lut, ddp);
    }

    auto int_end = chrono::system_clock::now();
    auto int_time = chrono::duration<float>(int_end - int_start);
    perfmon.record(PerformanceMonitor::ITERATION_INTEGRATION_TIME, (float)(int_time.count()));
    console_print(0, str(format("Repulsion and exchange matrix computed in: %0.3f seconds") % (float)(int_time.count())), CLIENT_SIM);
}

void print_csv_header(PerformanceMonitor &perfmon)
{
    cout << perfmon.str_csv_header();
}

void print_csv_data_all(Cfg_t &config, Lut_t &lut, PerformanceMonitor &perfmon)
{
    cout << perfmon.str_csv_data_all(config, lut);
}

void print_csv_data_average(Cfg_t &config, Lut_t &lut, PerformanceMonitor &perfmon)
{
    cout << perfmon.str_csv_data_average(config, lut);
}

int main(int argc, char *argv[])
{
    // Performance monitor object
    PerformanceMonitor perfmon;
    // Program config struct
    Cfg_t config;
    // Program lookup table values
    Lut_t lut;

    // Boost Program options
    parse_program_options(argc, argv, config);

    // Print the CSV header if requested, and exit the program
    if (config.enable_csv_header_output)
    {
        print_csv_header(perfmon);
        exit(EXIT_SUCCESS);
    }

    // Set the maximum number of thread to use for OMP and Eigen. This doesn't
    // appear to modulate the number of threads the LAPACK solver is using, so
    // something isn't right here. Leaving here anyways just in case I can fix
    // it sometime later.
    omp_set_dynamic(0); // Disable dynamic teams in OpenMP
    omp_set_num_threads(config.max_num_threads); // Set the number of maximum threads to use for OMP (Eigen will use this value)

    // Print the header
    print_header();

    // Fill out lut (the rest is done in populate_lookup_values once memory
    // has been allocated for the arrays inside)
    lut.matrix_dim = pow(config.num_partitions, 3.0);
    lut.step_size = (float)(4<<1)/(float)(config.num_partitions - 1);
    lut.step_size_cubed = pow(lut.step_size, 3.0);

    // Print program configuration
    print_program_configurations(config, lut);

    // Get and print CUDA information
    config_cuda(config);

    // Matrix declarations
    EigenFloatMatrix_t laplacian_matrix;
    EigenFloatMatrix_t kinetic_matrix;
    EigenFloatMatrix_t attraction_matrix;
    EigenFloatColVectorMap_t repulsion_diagonal(nullptr, lut.matrix_dim, 1);
    EigenFloatColVectorMap_t exchange_diagonal(nullptr, lut.matrix_dim, 1);
    EigenFloatMatrix_t fock_matrix;
    // Orbital values
    EigenFloatColVectorMap_t orbital_values(nullptr, lut.matrix_dim, 1);
    // Eigenvectors and eigenvalues
    EigenFloatColVectorMap_t eigenvalues(nullptr, lut.matrix_dim, 1);
    EigenFloatMatrixMap_t eigenvectors(nullptr, lut.matrix_dim, 1);

    // For the repulsion, exchange, and orbital values, we will overlay the
    // Eigen objects over unified memory that is allocated via CUDA (if
    // enabled). Otherwise we will malloc data normally. This overlay is
    // performed using the Eigen::Map object, which allows one to map a pointer
    // to data allocated outside of Eigen to an Eigen object, like a matrix or
    // vector. kernel.h externs the three pointers that are used to store the
    // addresses and the following call will allocate memory for them. We will
    // also allocate memory for the lookup tables used.
    DynamicDataPointers_t ddp;
    int allocate_error = 0;

    allocate_error |= cpu_allocate_integration_memory(lut, ddp);
    allocate_error |= cpu_allocate_eigensolver_memory(lut, ddp);

    if (!allocate_error)
    {
        new (&orbital_values) EigenFloatColVectorMap_t(ddp.orbital_values_data, lut.matrix_dim, 1);
        new (&repulsion_diagonal) EigenFloatColVectorMap_t(ddp.repulsion_diagonal_data, lut.matrix_dim, 1);
        new (&exchange_diagonal) EigenFloatColVectorMap_t(ddp.exchange_diagonal_data, lut.matrix_dim, 1);
        new (&eigenvectors) EigenFloatMatrixMap_t(ddp.eigenvectors_data, lut.matrix_dim, lut.matrix_dim);
        new (&eigenvalues) EigenFloatColVectorMap_t(ddp.eigenvalues_data, lut.matrix_dim, 1);
        // Populate LUTs
        populate_lookup_values(config, lut);
    }
    else
    {
        console_print_err(0, "Something went horribly wrong when allocating memory, aborting", CLIENT_SIM);
        exit(EXIT_FAILURE);
    }

    // Instantiate matrices and vectors
    new (&laplacian_matrix) EigenFloatMatrix_t(lut.matrix_dim, lut.matrix_dim);
    new (&kinetic_matrix) EigenFloatMatrix_t(lut.matrix_dim, lut.matrix_dim);
    new (&attraction_matrix) EigenFloatMatrix_t(lut.matrix_dim, lut.matrix_dim);
    new (&fock_matrix) EigenFloatMatrix_t(lut.matrix_dim, lut.matrix_dim);

    // Trimmed eigenvectors and eigenvalues
    EigenFloatColVector_t trimmed_eigenvalues(config.num_solutions, 1);
    EigenFloatMatrix_t trimmed_eigenvectors(lut.matrix_dim, config.num_solutions);

    console_print_hr(0, CLIENT_SIM);
    console_print(0, "Simulation start!", CLIENT_SIM);
    console_print_hr(0, CLIENT_SIM);
    auto sim_start = chrono::system_clock::now();

    // generate the second order Laplacian matrix for 3D space
    console_print(1, "Generating Laplacian matrix", CLIENT_SIM);
    generate_laplacian_matrix(lut, laplacian_matrix.data());
    // generate the kinetic energy matrix
    console_print(1, "Generating kinetic energy matrix", CLIENT_SIM);
    kinetic_matrix = (laplacian_matrix/(2.0*lut.step_size*lut.step_size));
    // generate the Coulombic attraction matrix
    console_print(1, "Generating electron-nucleus Coulombic attraction matrix", CLIENT_SIM);
    generate_attraction_matrix(lut, config.atomic_structure, attraction_matrix.data());

    // Main HF loop
    float last_total_energy = 0;
    float total_energy;
    float total_energy_percent_diff;
    int interation_count = 0;

    // initial solution
    fock_matrix = -kinetic_matrix - attraction_matrix;

    console_print(1, "Solving for initial solution", CLIENT_SIM);
    // Solve for Fock matrix eigenvectors
    eigenvectors = fock_matrix;
    eigensolver(config, lut, perfmon, ddp);
    orbital_values = eigenvectors.col(0);

    do
    {
        console_print_hr(0, CLIENT_SIM);
        console_print(0, str(format("Iteration Start: %d") % interation_count), CLIENT_SIM);
        console_print_hr(0, CLIENT_SIM);

        auto iteration_start = chrono::system_clock::now();

        // Generate repulsion and integraion matrices
        generate_repulsion_and_integration_matrices(config, lut, perfmon, ddp);

        // form fock matrix
        console_print(1, "Generating Fock matrix", CLIENT_SIM);
        fock_matrix = -kinetic_matrix - attraction_matrix + 2.0 * EigenFloatMatrix_t(repulsion_diagonal.asDiagonal()) - EigenFloatMatrix_t(exchange_diagonal.asDiagonal());

        // Solve for Fock matrix eigenvectors and eigenvalues
        eigenvectors = fock_matrix;
        eigensolver(config, lut, perfmon, ddp);
        orbital_values = eigenvectors.col(0);

        // Extract num_solutions eigenvalues
        trimmed_eigenvalues = eigenvalues.block(0, 0, config.num_solutions, 1);
        // Extract num_solutions eigenvectors
        trimmed_eigenvectors = eigenvectors.block(0, 0, lut.matrix_dim, config.num_solutions);

        total_energy = calculate_total_energy(orbital_values, kinetic_matrix, attraction_matrix, repulsion_diagonal, exchange_diagonal);
        perfmon.record(PerformanceMonitor::ITERATION_TOTAL_ENERGY, total_energy);
        total_energy_percent_diff = abs((total_energy - last_total_energy)/((total_energy + last_total_energy) / 2.0));

        console_print_hr(0, CLIENT_SIM);
        console_print(0, str(format("Total energy: %.3f") % (total_energy)), CLIENT_SIM);
        console_print(0, str(format("Energy %% diff: %.3f%%") % (total_energy_percent_diff * 100.0)), CLIENT_SIM);

        // update last value
        last_total_energy = total_energy;
        // update iteration count
        interation_count++;

        auto iteration_end = chrono::system_clock::now();
        auto iteration_time = chrono::duration<float>(iteration_end - iteration_start);
        perfmon.record(PerformanceMonitor::ITERATION_TOTAL_TIME, (float)(iteration_time.count()));
        console_print(0, str(format("Iteration end! Iteration time: %0.3f seconds") % (float)(iteration_time.count())), CLIENT_SIM);

        // Increment performance monitor data
        perfmon.next_iteration();

        // check if we've hit the maximum iteration limit
        if (interation_count == config.max_iterations)
        {
            break;
        }

        // check if we meet convergence condition
        if (abs(total_energy_percent_diff) < (config.convergence_percentage/100.0))
        {
            break;
        }
    }
    while(1);

    cpu_free_integration_memory(lut, ddp);
    cpu_free_eigensolver_memory(ddp);

    console_print_hr(0, CLIENT_SIM);
    console_print(0, "Final Eigenvalues:", CLIENT_SIM);
    stringstream ss;
    ss << trimmed_eigenvalues.transpose();
    console_print(0, ss.str(), CLIENT_SIM);
    ss.str(string()); // clear ss
    console_print_hr(0, CLIENT_SIM);
    console_print(0, str(format("Final Total energy: %.3f") % (total_energy)), CLIENT_SIM);
    console_print_hr(0, CLIENT_SIM);

    auto sim_end = chrono::system_clock::now();
    auto sim_time = chrono::duration<float>(sim_end - sim_start);
    perfmon.total_time = (float)(sim_time.count());
    console_print(0, str(format("Simulation end! Total time: %0.3f seconds") % (float)(sim_time.count())), CLIENT_SIM);

    console_print_hr(0, CLIENT_SIM);
    console_print(0, "Performance monitor records:", CLIENT_SIM);
    console_print(0, perfmon.str(), CLIENT_SIM);
    console_print_hr(0, CLIENT_SIM);

    if (config.enable_csv_data_all_output)
    {
        print_csv_data_all(config, lut, perfmon);
    }
    else if (config.enable_csv_data_average_output)
    {
        print_csv_data_average(config, lut, perfmon);
    }

    return 0;
}

