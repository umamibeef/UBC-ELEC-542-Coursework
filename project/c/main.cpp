#include <iostream>
#include <gsl/gsl_linalg.h>
#include <Eigen/Dense>
#include <boost/format.hpp>

#include "kernel.h"

int main(int argc, char ** argv)
{
    // number of partitions and limits
    int n = 11;
    int matrix_dim = n*n*n;
    int lim = 4;
    double h = (double)(4<<1)/(double)(n - 1);

    std::cout << boost::format("n = %d\n") % n;
    std::cout << boost::format("matrix_dim = %d\n") % matrix_dim;
    std::cout << boost::format("lim = %d\n") % lim;
    std::cout << boost::format("h = %f\n") % h;

    // coordinates
    double *coords_x;
    double *coords_y;
    double *coords_z;
    // matrices
    double *laplacian_matrix_data;
    double *kinetic_matrix_data;
    double *attraction_matrix_data;
    double *repulsion_matrix_data;
    double *exchange_matrix_data;
    double *fock_matrix_data;

    // allocate coordinate matrices and populate them
    coords_x = (double*)calloc(n, sizeof(double));
    coords_y = (double*)calloc(n, sizeof(double));
    coords_z = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        coords_x[i] = (double)(-lim) + (double)(i*h);
        coords_y[i] = (double)(-lim) + (double)(i*h);
        coords_z[i] = (double)(-lim) + (double)(i*h);
    }

    // allocate matrices
    laplacian_matrix_data = (double*)calloc(matrix_dim, sizeof(double));
    kinetic_matrix_data = (double*)calloc(matrix_dim, sizeof(double));
    attraction_matrix_data = (double*)calloc(matrix_dim, sizeof(double));
    repulsion_matrix_data = (double*)calloc(matrix_dim, sizeof(double));
    exchange_matrix_data = (double*)calloc(matrix_dim, sizeof(double));
    fock_matrix_data = (double*)calloc(matrix_dim, sizeof(double));

    // create gsl matrix views
    gsl_matrix_view laplacian = gsl_matrix_view_array(laplacian_matrix_data, matrix_dim, matrix_dim);
    gsl_matrix_view kinetic_energy = gsl_matrix_view_array(kinetic_matrix_data, matrix_dim, matrix_dim);
    gsl_matrix_view attraction = gsl_matrix_view_array(attraction_matrix_data, matrix_dim, matrix_dim);
    gsl_matrix_view repulsion = gsl_matrix_view_array(repulsion_matrix_data, matrix_dim, matrix_dim);
    gsl_matrix_view exchange = gsl_matrix_view_array(exchange_matrix_data, matrix_dim, matrix_dim);
    gsl_matrix_view fock = gsl_matrix_view_array(fock_matrix_data, matrix_dim, matrix_dim);

    // Eigen example
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

#if 0
    // GSL example
    double a_data[] = { 0.18, 0.60, 0.57, 0.96,
                        0.41, 0.24, 0.99, 0.58,
                        0.14, 0.30, 0.97, 0.66,
                        0.51, 0.13, 0.19, 0.85 };
    double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
    gsl_matrix_view m = gsl_matrix_view_array(a_data, 4, 4);
    gsl_vector_view b = gsl_vector_view_array(b_data, 4);
    gsl_vector *x = gsl_vector_alloc(4);
    int s;
    gsl_permutation * p = gsl_permutation_alloc(4);
    gsl_linalg_LU_decomp(&m.matrix, p, &s);
    gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);
    printf("x = \n");
    gsl_vector_fprintf(stdout, x, "%g");
    gsl_permutation_free(p);
    // Call CUDA example
    cuda_example();
#endif // 0

    // free memory
    free(coords_x);
    free(coords_y);
    free(coords_z);
    free(laplacian_matrix_data);
    free(kinetic_matrix_data);
    free(attraction_matrix_data);
    free(repulsion_matrix_data);
    free(exchange_matrix_data);
    free(fock_matrix_data);

    return 0;
}