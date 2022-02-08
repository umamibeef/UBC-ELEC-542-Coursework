"""
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
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import math
import functools
import time

numpy.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

# Matplotlib export settings
if False:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.size": 10 ,
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

# program constants
H2_BOND_LENGTH_ATOMIC_UNITS = 1.39839733222307
TINY_NUMBER = 1
IDX_X = 0
IDX_Y = 1
IDX_Z = 2

def main():

    # number of partitions in the solution
    N = 25

    # number of eigenvalues to calculate
    k = 6

    # convergence condition percentage
    convergence_percentage = 1

    # energy level to converge on
    energy_level = 0

    # generate coordinates
    coords = generate_coordinates(-2, 2, N)

    # calculate partition size
    h = coords[IDX_X][1]-coords[IDX_X][0]

    # generate attraction matrix for hydrogen molecule
    attraction_matrix_hydrogen = attraction_matrix_gen(attraction_func_hydrogen, N, coords)

    # generate attraction matrix for helium molecule
    attraction_matrix_helium = attraction_matrix_gen(attraction_func_helium, N, coords)

    # generate laplacian matrix
    laplacian_matrix = second_order_laplacian_3d_sparse_matrix_gen(N)

    # generate kinetic matrix
    kinetic_energy_matrix = (-1.0/(2.0*h**2))*laplacian_matrix

    # create base solution
    solution = scipy.sparse.csr_matrix((N**3, N**3))

    # last eigenvals
    last_total_energy = 0

    # first iteration
    first_iteration = True

    # iteration counter
    iteration_count = 0

    # total time
    total_time_start = time.time()

    while True:

        # iteration time
        iteration_time_start = time.time()

        if first_iteration:
            print('First iteration, zeros used as first guess')
            first_iteration = False
        else:

            print('Iteration: %d' % iteration_count)

        # create integration matrix
        integration_matrix = integration_matrix_gen(solution, energy_level, N, coords)

        # create Fock matrix
        fock_matrix = kinetic_energy_matrix + attraction_matrix_hydrogen + integration_matrix
        # fock_matrix = kinetic_energy_matrix + attraction_matrix_helium + integration_matrix
        # print('Fock is hermitian? ' + str(is_hermitian(fock_matrix)))

        # get eigenvectors and eigenvalues
        eigenvals, solution = scipy.sparse.linalg.eigsh(fock_matrix, k=k, which='SM')

        print('Solutions:')
        print('Fock: ' + str(fock_matrix.toarray()))
        print('Eigenvectors: ' + str(solution))
        print('Eigenvals: ' + str(eigenvals))

        # check percentage difference between previous and current eigenvalues
        total_energy = numpy.sum(numpy.absolute(eigenvals))

        # calculate total energy
        total_energy_percent_diff = abs(total_energy - last_total_energy)/((total_energy + last_total_energy) / 2)

        # check if we meet convergence condition
        if total_energy_percent_diff < (convergence_percentage/100.0):
            break

        print('Total energy %% diff: %.3f%%' % (total_energy_percent_diff * 100.0))

        # update last value
        last_total_energy = total_energy

        # update iteration count
        iteration_count = iteration_count + 1

        print("--- Iteration time: %s seconds ---" % (time.time() - iteration_time_start))

    print("--- Total time: %s seconds ---" % (time.time() - total_time_start))

    # plot data
    make_plots(N, coords, solution)

# This functions returns whether or not the matrix is symmetric
def is_symmetric(A, tol=1e-8):
    return scipy.sparse.linalg.norm(A-A.T, scipy.Inf) < tol;

# This functions returns whether or not the matrix is symmetric
def is_hermitian(A, tol=1e-8):
    return scipy.sparse.linalg.norm(A-A.H, scipy.Inf) < tol;


#
# This function takes in a matrix index (row or column) and returns the
# associated coordinate indices as a tuple.
#
def matrix_index_to_coordinate_indices(matrix_index, N):

    # Z is kind of like the MSB, as it changes less often, so we'll treat this
    # as base conversion of sorts where to new base is N
    
    base_10_num = matrix_index
    digits = []

    # while we have a base 10 number to work with
    for i in range(3):
        # do moduluses to extract the digits of the new number
        digits = digits + [base_10_num % N]
        # go to the next digit
        base_10_num = base_10_num // N

    return tuple(digits)

#
# This function coverts coordinate indices to a matrix index
#
def coordinate_indices_to_matrix_index(coord_indices, N):

    return coord_indices[IDX_X] + coord_indices[IDX_Y]*N + coord_indices[IDX_Z]*N*N 
#
# This function takes in coordinate indices and returns the coordinates
# associated with them.
#
def coordinate_index_to_coordinates(coord_indices, coords):

    return (coords[IDX_X][coord_indices[IDX_X]], coords[IDX_Y][coord_indices[IDX_Y]], coords[IDX_Z][coord_indices[IDX_Z]])

#
# Generate the coordinates for the solution space, with the assumption that it
# is cubic.
#
def generate_coordinates(minimum, maximum, N):

    x = numpy.linspace(minimum, maximum, N)
    y = numpy.linspace(minimum, maximum, N)
    z = numpy.linspace(minimum, maximum, N)

    return (x, y, z)

#
# This function returns the nuclear attraction for an electron in the Helium
# element simulation. A small number TINY_NUMBER is provided to prevent divide
# by zero scenarios.
#
def attraction_func_helium(coords):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    denominator = math.sqrt(x**2 + y**2 + z**2)

    return -((2.0/(TINY_NUMBER + denominator)))

#
# This function returns the nuclear attraction for an electron in the Hydrogen
# molecule simulation. A small number TINY_NUMBER is provided to prevent divide
# by zero scenarios.
#
def attraction_func_hydrogen(coords):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    denominator_1 = math.sqrt(((H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + z**2)
    denominator_2 = math.sqrt(((-H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + z**2)

    return -((1.0/(TINY_NUMBER + denominator_1)) + (1.0/(TINY_NUMBER + denominator_2)))

#
# This functions calculates the repulsion between two electrons. A small number
# TINY_NUMBER is provided to prevent divide by zero scenarios.
#
@functools.lru_cache
def repulsion_func(coords_1, coords_2):

    x1 = coords_1[IDX_X]
    y1 = coords_1[IDX_Y]
    z1 = coords_1[IDX_Z]

    x2 = coords_2[IDX_X]
    y2 = coords_2[IDX_Y]
    z2 = coords_2[IDX_Z]

    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return ((1.0/(TINY_NUMBER + denominator)))

#
# This function generates an attraction matrix with the specified attraction
# function and coordinates
#
def attraction_matrix_gen(attraction_func, N, coords):

    # use scipy sparse matrix generation
    # create the diagonal 
    diagonal = [attraction_func(coordinate_index_to_coordinates(matrix_index_to_coordinate_indices(i, N), coords)) for i in range(N**3)]

    # now generate the matrix with the desired diagonal
    matrix = scipy.sparse.spdiags(data=diagonal, diags=0, m=N**3, n=N**3)

    return matrix.tocoo()

#
# This function generates the second order laplacian matrix for 3D space for N
# partitions.
#
#   e.g. N = 3
#   
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |-6 1   | 1     |       | 1     |       |       |       |       |       | 
#     | 1-6 1 |   1   |       |   1   |       |       |       |       |       | 
#     |   1-6 |     1 |       |     1 |       |       |       |       |       | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     | 1     |-6 1   | 1     |       | 1     |       |       |       |       | 
#     |   1   | 1-6 1 |   1   |       |   1   |       |       |       |       | 
#     |     1 |   1-6 |     1 |       |     1 |       |       |       |       | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       | 1     |-6 1   |       |       | 1     |       |       |       | 
#     |       |   1   | 1-6 1 |       |       |   1   |       |       |       | 
#     |       |     1 |   1-6 |       |       |     1 |       |       |       | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     | 1     |       |       |-6 1   | 1     |       | 1     |       |       | 
#     |   1   |       |       | 1-6 1 |   1   |       |   1   |       |       | 
#     |     1 |       |       |   1-6 |     1 |       |     1 |       |       | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       | 1     |       | 1     |-6 1   | 1     |       | 1     |       |
#     |       |   1   |       |   1   | 1-6 1 |   1   |       |   1   |       |
#     |       |     1 |       |     1 |   1-6 |     1 |       |     1 |       |
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       |       | 1     |       | 1     |-6 1   |       |       | 1     | 
#     |       |       |   1   |       |   1   | 1-6 1 |       |       |   1   | 
#     |       |       |     1 |       |     1 |   1-6 |       |       |     1 | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       |       |       | 1     |       |       |-6 1   | 1     |       | 
#     |       |       |       |   1   |       |       | 1-6 1 |   1   |       | 
#     |       |       |       |     1 |       |       |   1-6 |     1 |       | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       |       |       |       | 1     |       | 1     |-6 1   | 1     | 
#     |       |       |       |       |   1   |       |   1   | 1-6 1 |   1   | 
#     |       |       |       |       |     1 |       |     1 |   1-6 |     1 | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
#     |       |       |       |       |       | 1     |       | 1     |-6 1   | 
#     |       |       |       |       |       |   1   |       |   1   | 1-6 1 | 
#     |       |       |       |       |       |     1 |       |     1 |   1-6 | 
#     +-------+-------+-------+-------+-------+-------+-------+-------+-------+
def second_order_laplacian_3d_sparse_matrix_gen(N):

    # create block matrices
    # sub main block diagonals (containing -6 surrounded by 1s) found within N*N blocks
    # e.g N=3
    # +-------+
    # |-6 1   |
    # | 1-6 1 |
    # |   1-6 |
    # +-------+
    sub_main_block_diag = [-6.0 for i in range(N)]
    sub_main_block_outer_diags = [1.0 for i in range(N)]
    sub_main_block = scipy.sparse.spdiags(data=[sub_main_block_outer_diags, sub_main_block_diag, sub_main_block_outer_diags], diags=[-1,0,1], m=N, n=N)

    # create mini identity blocks found within N*N blocks
    # e.g N=3
    # +-------+
    # | 1     |
    # |   1   |
    # |     1 |
    # +-------+
    mini_ident_block = scipy.sparse.identity(N)

    # create a list of blocks to be used in the sparse.bmat function to generate
    # main block along the diagonal
    # e.g. N=3
    # +-------+-------+-------+
    # |-6 1   | 1     |       |
    # | 1-6 1 |   1   |       |
    # |   1-6 |     1 |       |
    # +-------+-------+-------+
    # | 1     |-6 1   | 1     |
    # |   1   | 1-6 1 |   1   |
    # |     1 |   1-6 |     1 |
    # +-------+-------+-------+
    # |       | 1     |-6 1   |
    # |       |   1   | 1-6 1 |
    # |       |     1 |   1-6 |
    # +-------+-------+-------+
    block_lists = [[None for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            # add main blocks
            if j == i:
                block_lists[i][j] = sub_main_block
            # add diagonal blocks
            if (j == (i + 1)) and ((i + 1) != N):
                block_lists[i][j] = mini_ident_block
            if (j == (i - 1)) and ((i - 1) != -1):
                block_lists[i][j] = mini_ident_block
    main_block = scipy.sparse.bmat(block_lists)

    # create large identity blocks of size N*N
    # e.g. N=3
    # +-------+-------+-------+
    # | 1     |       |       |
    # |   1   |       |       |
    # |     1 |       |       |
    # +-------+-------+-------+
    # |       | 1     |       |
    # |       |   1   |       |
    # |       |     1 |       |
    # +-------+-------+-------+
    # |       |       | 1     |
    # |       |       |   1   |
    # |       |       |     1 |
    # +-------+-------+-------+
    large_ident_block = scipy.sparse.identity(N*N)

    # create a list of blocks to be used in the sparse.bmat function to generate
    # the final laplacian_matrix
    block_lists = [[None for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            # add main blocks
            if j == i:
                block_lists[i][j] = main_block
            # add diagonal blocks
            if (j == (i + 1)) and ((i + 1) != N):
                block_lists[i][j] = large_ident_block
            if (j == (i - 1)) and ((i - 1) != -1):
                block_lists[i][j] = large_ident_block

    return scipy.sparse.bmat(block_lists)
 
#
# This function evaluates the integration function used by both the Helium
# element and the Hydrogen molecule.
#
def integration_func(eigenvectors, eigenvector_index, N, coords_1, all_coords, h):

    # running sum
    sum = 0

    # calculate the integration over the solution space of the specified psi squared
    for xi, x in enumerate(all_coords[IDX_X]):
        for yi, y in enumerate(all_coords[IDX_Y]):
            for zi, z in enumerate(all_coords[IDX_Z]):
                # first integration weight is 0
                if xi == 0 or yi == 0 or zi == 0:
                    w = 0
                else:
                    w = h
                matrix_index = xi + yi*N + zi*N*N
                coords_2 = (x, y, z)
                sum = sum + w*(eigenvectors[:,eigenvector_index][matrix_index])**2*repulsion_func(coords_1, coords_2)

    return sum

# This function generates the the integration matrix
def integration_matrix_gen(eigenvectors, eigenvector_index, N, coords):

    # extract h from coordinates
    h = coords[IDX_X][1] - coords[IDX_X][0]

    # turn eigenvectors into a normal array if necessary
    if type(eigenvectors) != numpy.ndarray:
        eigenvectors = eigenvectors.toarray()

    # use scipy sparse matrix generation
    # create the diagonal 
    diagonal = [integration_func(eigenvectors, eigenvector_index, N, coordinate_index_to_coordinates(matrix_index_to_coordinate_indices(i, N), coords), coords, h) for i in range(N**3)]

    # now generate the matrix with the desired diagonal
    matrix = scipy.sparse.spdiags(data=diagonal, diags=0, m=N**3, n=N**3)

    return matrix.tocoo()

def make_plots(N, coords, solution):

    # With help from this SO answer
    # https://stackoverflow.com/a/66939879

    # reshape solution for lowest energy level
    energy_0_points = numpy.square(solution[:,0].reshape((N,N,N)).transpose())

    x_start, x_end = (coords[IDX_X][0],coords[IDX_X][-1])
    y_start, y_end = (coords[IDX_Y][0],coords[IDX_Y][-1])
    z_start, z_end = (coords[IDX_Z][0],coords[IDX_Z][-1])

    # look at z-plane
    fig = plt.figure()
    axes = plt.axes(projection="3d")
    axes.xaxis.pane.fill = False
    axes.yaxis.pane.fill = False
    axes.zaxis.pane.fill = False

    axes.set_xlim3d(x_start, x_end)
    axes.set_ylim3d(y_start, y_end)
    axes.set_zlim3d(z_start, z_end)

    # create a mask for the data to make the visualization clearer
    mask = energy_0_points > (energy_0_points.max() * 0.2)
    idx = numpy.arange(int(numpy.prod(energy_0_points.shape)))
    x, y, z = numpy.unravel_index(idx, energy_0_points.shape)
    x, y, z = numpy.meshgrid(coords[IDX_X], coords[IDX_Y], coords[IDX_Z])
    axes.scatter(x, y, z, c=energy_0_points.flatten(), s=200.0 * mask, edgecolor='face', alpha=0.2, marker='o', cmap='viridis', linewidth=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
