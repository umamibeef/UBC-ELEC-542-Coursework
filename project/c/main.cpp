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
#define OMP_NUM_THREADS (16)
#include <omp.h>

// CUDA
#include <cuda.h>

// Program includes
#include "main.hpp"
#include "config.hpp"
#include "console.hpp"
#include "kernel.h"
#include "version.hpp"

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

    console_print(0, ss.str(), SIM);
}

void populate_lookup_values(Cfg_t &config)
{
    // Resize LUTs
    config.coordinate_value_array.resize(IDX_NUM);
    config.coordinate_index_array.resize(IDX_NUM);
    for (int i = 0; i < IDX_NUM; i++)
    {
        config.coordinate_value_array[i].resize(config.matrix_dim);
        config.coordinate_index_array[i].resize(config.matrix_dim);
    }

    // Populate LUTs
    for (int coordinate_index = 0; coordinate_index < config.matrix_dim; coordinate_index++)
    {
        // convert coordinate index to x,y,z coordinate indices
        int base_10_num = coordinate_index;
        for (int i = 0; i < IDX_NUM; i++)
        {
            config.coordinate_index_array[i][coordinate_index] = base_10_num % config.num_partitions;
            base_10_num /= config.num_partitions;

            config.coordinate_value_array[i][coordinate_index] = (float)(-config.limit) + ((float)config.coordinate_index_array[i][coordinate_index]*config.step_size);
        }
    }

    // Calculate step_size_cubed
    config.step_size_cubed = pow(config.step_size, 3.0);
}

void linear_coordinate_index_to_spatial_coordinates_index(Cfg_t &config, int coordinate_index, int * coordinate_index_array)
{
    // convert coordinate index to x,y,z coordinate indices
    int base_10_num = coordinate_index;
    for (int i = 0; i < IDX_NUM; i++)
    {
        coordinate_index_array[i] = base_10_num % config.num_partitions;
        base_10_num /= config.num_partitions;
    }
}

void linear_coordinate_index_to_spatial_coordinates_values(Cfg_t &config, int coordinate_index, float *coordinate_value_array)
{
    int coordinates_indices[IDX_NUM];
    linear_coordinate_index_to_spatial_coordinates_index(config, coordinate_index, coordinates_indices);
    coordinate_value_array[IDX_X] = (float)(-config.limit) + ((float)coordinates_indices[IDX_X]*config.step_size);
    coordinate_value_array[IDX_Y] = (float)(-config.limit) + ((float)coordinates_indices[IDX_Y]*config.step_size);
    coordinate_value_array[IDX_Z] = (float)(-config.limit) + ((float)coordinates_indices[IDX_Z]*config.step_size);
}

// Generate the 3D Laplacian matrix for the given number of partitions
void generate_laplacian_matrix(Cfg_t &config, float *matrix)
{
    // Set matrix to 0
    memset(matrix,0.0,config.matrix_dim*config.matrix_dim*sizeof(float));

    int col_index_x;
    int col_index_y;
    int col_index_z;
    int row_index_x;
    int row_index_y;
    int row_index_z;

    for (int row_coordinate_index = 0; row_coordinate_index < config.matrix_dim; row_coordinate_index++)
    {
        for (int col_coordinate_index = 0; col_coordinate_index < config.matrix_dim; col_coordinate_index++)
        {
            col_index_x = config.coordinate_index_array[IDX_X][col_coordinate_index];
            col_index_y = config.coordinate_index_array[IDX_Y][col_coordinate_index];
            col_index_z = config.coordinate_index_array[IDX_Z][col_coordinate_index];

            row_index_x = config.coordinate_index_array[IDX_X][row_coordinate_index];
            row_index_y = config.coordinate_index_array[IDX_Y][row_coordinate_index];
            row_index_z = config.coordinate_index_array[IDX_Z][row_coordinate_index];

            // U(x,y,z)
            if (row_coordinate_index == col_coordinate_index)
            {
                matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = -6.0;
            }

            if ((row_index_y == col_index_y) && (row_index_z == col_index_z))
            {
                // U(x-1,y,z)
                if (row_index_x == col_index_x + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
                // U(x+1,y,z)
                if (row_index_x == col_index_x - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
            }

            if ((row_index_x == col_index_x) && (row_index_z == col_index_z))
            {
                // U(x,y-1,z)
                if (row_index_y == col_index_y + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
                // U(x,y+1,z)
                if (row_index_y == col_index_y - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
            }

            if ((row_index_x == col_index_x) && (row_index_y == col_index_y))
            {
                // U(x,y,z-1)
                if (row_index_z == col_index_z + 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
                // U(x,y,z+1)
                if (row_index_z == col_index_z - 1)
                {
                    matrix[row_coordinate_index + col_coordinate_index*config.matrix_dim] = 1.0;
                }
            }
        }
    }
}

// Helium atom nucleus-electron Coulombic attraction function
float attraction_function_helium(Cfg_t &config, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float x = config.coordinate_value_array[IDX_X][linear_coordinates];
    float y = config.coordinate_value_array[IDX_Y][linear_coordinates];
    float z = config.coordinate_value_array[IDX_Z][linear_coordinates];

    float denominator = sqrt(pow(x, 2.0) + pow(y + Y_OFFSET, 2.0) + pow(z + Z_OFFSET, 2.0));

    if (abs(denominator) < epsilon)
    {
        denominator = sqrt(TINY_NUMBER);
    }

    return (2.0/(denominator));
}

// Hydrogen molecule nucleus-electron Coulombic attraction function
float attraction_function_hydrogen(Cfg_t &config, int linear_coordinates)
{
    const float epsilon = EPSILON;

    float x = config.coordinate_value_array[IDX_X][linear_coordinates];
    float y = config.coordinate_value_array[IDX_Y][linear_coordinates];
    float z = config.coordinate_value_array[IDX_Z][linear_coordinates];

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
void generate_attraction_matrix(Cfg_t &config, atomic_structure_e atomic_structure, float *matrix)
{
    // Set matrix to 0
    memset(matrix,0.0,config.matrix_dim*config.matrix_dim*sizeof(float));

    if (atomic_structure == HELIUM_ATOM)
    {
        for (int diagonal_index = 0; diagonal_index < config.matrix_dim; diagonal_index++)
        {
            matrix[diagonal_index + diagonal_index*config.matrix_dim] = attraction_function_helium(config, diagonal_index);
        }
    }
    else if (atomic_structure == HYDROGEN_MOLECULE)
    {
        for (int diagonal_index = 0; diagonal_index < config.matrix_dim; diagonal_index++)
        {
            matrix[diagonal_index + diagonal_index*config.matrix_dim] = attraction_function_hydrogen(config, diagonal_index);
        }
    }
}

// Electron-electron Coulombic repulsion function
float repulsion_function(Cfg_t &config, int linear_coordinates_1, int linear_coordinates_2)
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

float repulsion_matrix_integrand_function(Cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return pow(orbital_values[linear_coords_2], 2.0)*repulsion_function(config, linear_coords_1, linear_coords_2);
}

float exchange_matrix_integrand_function(Cfg_t &config, float *orbital_values, int linear_coords_1, int linear_coords_2)
{
    return orbital_values[linear_coords_1]*orbital_values[linear_coords_2]*repulsion_function(config, linear_coords_1, linear_coords_2);
}

void generate_repulsion_matrix(Cfg_t &config, float *orbital_values, float *matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += repulsion_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*config.step_size_cubed;
    }
}


void generate_exchange_matrix(Cfg_t &config, float *orbital_values, float *matrix)
{
    for (int electron_one_coordinate_index = 0; electron_one_coordinate_index < config.matrix_dim; electron_one_coordinate_index++)
    {
        float sum = 0;
        for (int electron_two_coordinate_index = 0; electron_two_coordinate_index < config.matrix_dim; electron_two_coordinate_index++)
        {
            sum += exchange_matrix_integrand_function(config, orbital_values, electron_one_coordinate_index, electron_two_coordinate_index);
        }
        matrix[electron_one_coordinate_index + electron_one_coordinate_index*config.matrix_dim] = sum*config.step_size_cubed;
    }
}

template <typename A, typename B, typename C, typename D, typename E>
float calculate_total_energy(Eigen::MatrixBase<A> &orbital_values, Eigen::MatrixBase<B> &kinetic_matrix, Eigen::MatrixBase<C> &attraction_matrix, Eigen::MatrixBase<D> &repulsion_matrix, Eigen::MatrixBase<E> &exchange_matrix)
{ 
    // orbital values are real, so no need to take conjugate
    EigenFloatRowVector_t psi_prime = orbital_values.transpose();
    EigenFloatColVector_t psi = orbital_values;

    float energy_sum = 0;

    // sum for the total number of electrons in the systems: 2
    for (int i = 0; i < 2; i++)
    {
        auto hermitian_term = psi_prime*(-kinetic_matrix - attraction_matrix)*psi;
        auto hf_term = 0.5*psi_prime*(2*repulsion_matrix - exchange_matrix)*psi;
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
// @param      config       The program config struct
// @param      matrix       The matrix originally containing the matrix to be
//                          solved and the resulting eigenvectors
// @param      eigenvalues  The eigenvalues of the solution
//
// @return     True if the solver found solutions, false if the solver failed
//             for some reason
//
bool lapack_solve_eigh(Cfg_t &config, float *matrix, float *eigenvalues)
{
    char jobz = 'V'; // compute eigenvalues and eigenvectors.
    char uplo = 'U'; // perform calculation on upper triangle of matrix
    lapack_int n = config.matrix_dim; // order of the matrix (size)
    lapack_int lda = config.matrix_dim; // the leading dimension of the array A. LDA >= max(1,N).
    lapack_int info = 0;
    lapack_int liwork;
    lapack_int lwork;
    lapack_int* iwork = nullptr;
    float* work = nullptr;

    console_print(2, TAB1 "LAPACK solver debug", LAPACK);
    console_print(2, str(format(TAB2 "n = %d") % n), LAPACK);
    console_print(2, str(format(TAB2 "lda = %d") % lda), LAPACK);

    // Setting lwork and liwork manually based on the LAPACK documentation
    lwork = 1 + 6*n + 2*n*n;
    liwork = 3 + 5*n;

    console_print(2, str(format(TAB2 "liwork = %d") % liwork), LAPACK);
    console_print(2, str(format(TAB2 "lwork = %d") % lwork), LAPACK);

    // Allocate memory for work arrays
    iwork = (lapack_int*)LAPACKE_malloc(sizeof(lapack_int) * liwork);
    if(iwork == nullptr)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print(2, TAB2 "FATAL! Could not allocate iwork array", LAPACK);
    }
    work = (float*)LAPACKE_malloc(sizeof(float) * lwork);
    if(work == nullptr)
    {
        info = LAPACK_WORK_MEMORY_ERROR;
        console_print(2, TAB2 "FATAL! Could not allocate work array", LAPACK);
    }

    // Call LAPACK function and adjust info if our work areas are OK
    if ((iwork != nullptr) && (work != nullptr))
    {
        console_print(2, TAB2 "calling LAPACK function", LAPACK);
        LAPACK_ssyevd(&jobz, &uplo, &n, matrix, &lda, eigenvalues, work, &lwork, iwork, &liwork, &info);
        if( info < 0 ) {
            info = info - 1;
        }
    }

    // Release memory and exit
    LAPACKE_free(iwork);
    LAPACKE_free(work);

    console_print(2, str(format(TAB2 "info = %d") % info), LAPACK);

    return (info==0);
}

int main(int argc, char *argv[])
{
    // Print the header
    print_header();

    // Set the maximum number of thread to use for OMP and Eigen
    omp_set_num_threads(OMP_NUM_THREADS); // Set the number of maximum threads to use for OMP
    Eigen::setNbThreads(OMP_NUM_THREADS); // Set the number of maximum threads to use for Eigen

    // Boost Program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("iterations", po::value<int>()->default_value(50), "set maximum number of iterations")
        ("partitions", po::value<int>()->default_value(12), "set number of partitions to divide solution space")
        ("limit", po::value<int>()->default_value(4), "set the solution space maximum x=y=z limit")
        ("convergence", po::value<float>()->default_value(0.01), "set the convergence condition (%)")
        ("structure", po::value<int>()->default_value(0), "set the atomic structure: (0:He, 1:H2)")
        ("verbosity", po::value<int>()->default_value(0), "set the verbosity of the program")
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

    // Program configuration
    Cfg_t config;
    config.num_partitions = vm["partitions"].as<int>();
    config.limit = vm["limit"].as<int>();
    config.matrix_dim = config.num_partitions*config.num_partitions*config.num_partitions;
    config.step_size = (float)(4<<1)/(float)(config.num_partitions - 1);
    config.max_iterations = vm["iterations"].as<int>();
    config.num_solutions = 6;
    config.convergence_percentage = vm["convergence"].as<float>();
    program_verbosity = vm["verbosity"].as<int>();

    // Print program information
    console_print(0, "Program Configurations:\n", SIM);
    console_print(0, str(format(TAB1 "Iterations = %d") % config.max_iterations), SIM);
    console_print(0, str(format(TAB1 "Num Partitions = %d") % config.num_partitions), SIM);
    console_print(0, str(format(TAB1 "Matrix Dimension = %d") % config.matrix_dim), SIM);
    console_print(0, str(format(TAB1 "Limits = %d") % config.limit), SIM);
    console_print(0, str(format(TAB1 "Step Size = %f") % config.step_size), SIM);
    if (atomic_structure == HELIUM_ATOM)
    {
        console_print(0, TAB1 "Atomic Structure: Helium Atom", SIM);
    }
    else if (atomic_structure == HYDROGEN_MOLECULE)
    {
        console_print(0, TAB1 "Atomic Structure: Hydrogen Molecule", SIM);
    }
    console_print_spacer(0, SIM);
    // Print CUDA information
    cuda_print_device_info();

    // Populate LUTs
    populate_lookup_values(config);

    // Matrix declarations
    EigenFloatMatrix_t laplacian_matrix;
    EigenFloatMatrix_t kinetic_matrix;
    EigenFloatMatrix_t attraction_matrix;
    EigenFloatMatrixMap_t repulsion_matrix(nullptr, config.matrix_dim, config.matrix_dim);
    EigenFloatMatrixMap_t exchange_matrix(nullptr, config.matrix_dim, config.matrix_dim);
    EigenFloatMatrix_t fock_matrix;
    // Orbital values
    EigenFloatColVectorMap_t orbital_values(nullptr, config.matrix_dim, 1);
    // Eigenvectors and eigenvalues
    EigenFloatColVector_t eigenvalues;
    EigenFloatMatrix_t eigenvectors;

    // Instantiate matrices and vectors
    new (&laplacian_matrix) EigenFloatMatrix_t(config.matrix_dim, config.matrix_dim);
    new (&kinetic_matrix) EigenFloatMatrix_t(config.matrix_dim, config.matrix_dim);
    new (&attraction_matrix) EigenFloatMatrix_t(config.matrix_dim, config.matrix_dim);
    new (&fock_matrix) EigenFloatMatrix_t(config.matrix_dim, config.matrix_dim);
    new (&eigenvectors) EigenFloatMatrix_t(config.matrix_dim, config.matrix_dim);
    new (&eigenvalues) EigenFloatColVector_t(config.matrix_dim, 1);
    // For the repulsion, exchange, and orbital values, we will overlay the
    // Eigen objects over unified memory that is allocated via CUDA. This
    // overlay is performed using the Eigen::Map class, which allows one to map
    // a pointer to data allocated outside of Eigen to an Eigen object, like a
    // matrix or vector. kernel.h externs the three pointers that are used to
    // store the addresses and the following call will allocate memory for them.
    cuda_allocate_shared_memory(config);
    // make sure we got some good pointers out of that call
    if ((orbital_values_shared != nullptr) && (repulsion_matrix_shared != nullptr) && (exchange_matrix_shared != nullptr))
    {
        new (&orbital_values) EigenFloatColVectorMap_t(orbital_values_shared, config.matrix_dim, 1);
        new (&repulsion_matrix) EigenFloatMatrixMap_t(repulsion_matrix_shared, config.matrix_dim, config.matrix_dim);
        new (&exchange_matrix) EigenFloatMatrixMap_t(exchange_matrix_shared, config.matrix_dim, config.matrix_dim);
    }
    else
    {
        console_print_err(0, "Something went horribly wrong when allocating shared memory, aborting", SIM);
        exit(EXIT_FAILURE);
    }

    // Trimmed eigenvectors and eigenvalues
    EigenFloatColVector_t trimmed_eigenvalues(config.num_solutions, 1);
    EigenFloatMatrix_t trimmed_eigenvectors(config.matrix_dim, config.num_solutions);

    console_print(0, "Simulation start!", SIM);
    auto sim_start = chrono::system_clock::now();

    // generate the second order Laplacian matrix for 3D space
    console_print(1, "Generating kinetic energy matrix", SIM);
    generate_laplacian_matrix(config, laplacian_matrix.data());
    // generate the kinetic energy matrix
    kinetic_matrix = (laplacian_matrix/(2.0*config.step_size*config.step_size));
    // generate the Coulombic attraction matrix
    console_print(1, "Generating electron-nucleus Coulombic attraction matrix", SIM);
    generate_attraction_matrix(config, atomic_structure, attraction_matrix.data());

    // Main HF loop
    float last_total_energy = 0;
    float total_energy;
    float total_energy_percent_diff;
    int interation_count = 0;

    // initial solution
    fock_matrix = -kinetic_matrix - attraction_matrix;

    console_print(1, "Obtaining eigenvalues and eigenvectors for initial solution...", SIM);
    // LAPACK solver
    eigenvectors = fock_matrix;
    if (!lapack_solve_eigh(config, eigenvectors.data(), eigenvalues.data()))
    {
        console_print_err(0, "Something went horribly wrong with the solver, aborting", SIM);
        exit(EXIT_FAILURE);
    }
    orbital_values = eigenvectors.col(0);

    do
    {
        console_print(0, str(format("Iteration Start: %d") % interation_count), SIM);

        auto iteration_start = chrono::system_clock::now();

        // zero out matrices
        repulsion_matrix.Zero(config.matrix_dim, config.matrix_dim);
        exchange_matrix.Zero(config.matrix_dim, config.matrix_dim);

        // generate repulsion and exchange matrices on GPU
        cuda_numerical_integration_kernel(config, orbital_values.data(), repulsion_matrix.data(), exchange_matrix.data());

        // generate repulsion and exchange matrices on CPU
        // generate repulsion matrix
        console_print(1, "Generating electron-electron Coulombic repulsion matrix", SIM);
        generate_repulsion_matrix(config, orbital_values.data(), repulsion_matrix.data());
        // generate exchange matrix
        console_print(1, "Generating electron-electron exchange matrix", SIM);
        generate_exchange_matrix(config, orbital_values.data(), exchange_matrix.data());
        // form fock matrix
        console_print(1, "Generating Fock matrix", SIM);
        fock_matrix = -kinetic_matrix - attraction_matrix + 2.0*repulsion_matrix - exchange_matrix;

        console_print(1, "Obtaining eigenvalues and eigenvectors...", SIM);

        // LAPACK solver, the same one used in numpy
        eigenvectors = fock_matrix;
        if (!lapack_solve_eigh(config, eigenvectors.data(), eigenvalues.data()))
        {
            console_print(0, "Something went horribly wrong with the solver, aborting", SIM);
            exit(EXIT_FAILURE);
        }

        // Extract orbital_values
        orbital_values = eigenvectors.col(0);
        // Extract num_solutions eigenvalues
        trimmed_eigenvalues = eigenvalues.block(0, 0, config.num_solutions, 1);
        // Extract num_solutions eigenvectors
        trimmed_eigenvectors = eigenvectors.block(0, 0, config.matrix_dim, config.num_solutions);

        total_energy = calculate_total_energy(orbital_values, kinetic_matrix, attraction_matrix, repulsion_matrix, exchange_matrix);
        total_energy_percent_diff = abs((total_energy - last_total_energy)/((total_energy + last_total_energy) / 2.0));

        console_print(0, str(format("Total energy: %.3f") % (total_energy)), SIM);
        console_print(0, str(format("Energy %% diff: %.3f%%") % (total_energy_percent_diff * 100.0)), SIM);

        // update last value
        last_total_energy = total_energy;

        // update iteration count
        interation_count++;

        auto iteration_end = chrono::system_clock::now();
        auto iteration_time = chrono::duration<float>(iteration_end - iteration_start);

        console_print(0, str(format("Iteration end! Iteration time: %0.3f seconds") % (float)(iteration_time.count())), SIM);
        console_print_spacer(0, SIM);

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

    // free shared memory for GPU calculations
    cuda_free_shared_memory();

    console_print_spacer(0, SIM);
    console_print(0, "Final Eigenvalues:", SIM);
    stringstream ss;
    ss << trimmed_eigenvalues.transpose();
    console_print(0, ss.str(), SIM);
    ss.str(string()); // clear ss
    console_print_spacer(0, SIM);
    console_print(0, str(format("Final Total energy: %.3f") % (total_energy)), SIM);
    console_print_spacer(0, SIM);

    auto sim_end = chrono::system_clock::now();
    auto sim_time = chrono::duration<float>(sim_end - sim_start);
    console_print(0, str(format("Simulation end! Total time: %0.3f seconds") % (float)(sim_time.count())), SIM);

    return 0;
}