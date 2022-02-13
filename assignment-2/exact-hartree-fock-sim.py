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

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import math
import functools
import time
import pickle
import datetime # for timestamping
import multiprocessing # for multiprocessing (MP) of matrix generation
import tqdm # progress bar for MP

numpy.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: '%.3g' % x))

# Matplotlib export settings
if False:
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.size': 10 ,
        'font.family': 'serif',  # use serif/main font for text elements
        'text.usetex': True,     # use inline math for ticks
        'pgf.rcfonts': False     # don't setup fonts from rc parameters
    })

# program constants
H2_BOND_LENGTH_ATOMIC_UNITS = 1.39839733222307
TINY_NUMBER = 1e-6
IDX_X = 0
IDX_Y = 1
IDX_Z = 2
IDX_START = 0
IDX_END = 1
DATETIME_STR_FORMAT = '[%Y/%m/%d-%H:%M:%S]'
ENABLE_MP = True # multiprocessing

#
# This object encapsulates the results for pickling/unpickling
#
class Results:

    def __init__(self, eigenvectors=None, eigenvalues=None, args=None):

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.args = args
        self.iteration_times = []
        self.total_time = None

    def load(self, input_file):
        with open(input_file, 'rb') as file:
            tmp_dict = pickle.load(file)         

        self.__dict__.update(tmp_dict) 

    def save(self, output_file):
        with open(output_file, 'wb') as file:
            pickle.dump(self.__dict__, file, 2)

    def display_data(self):
        for key in self.__dict__:
            print(key)
            print(self.__dict__[key])

#
# This function generates the plots
#
def make_plots(results):

    # with help from this SO answer (4D scatter plots in matplotlib)
    # https://stackoverflow.com/a/66939879

    # number of partitions in the solution
    N = results.args.num_partitions
    # limits of sim
    limits = (-1*results.args.limit, results.args.limit)
    # generate coordinates
    coords = generate_coordinates(limits[IDX_START], limits[IDX_END], N)

    # eigenvectors
    eigenvectors = results.eigenvectors

    # total energy levels
    total_energy_levels = len(results.eigenvalues)

    # create figure object
    fig = plt.figure(figsize=(9,10), dpi=150, constrained_layout=True)

    # modify subject formatting
    if results.args.target_subject == 'he':
        title_subject = 'He'
    elif results.args.target_subject == 'h2':
        title_subject = 'H_2'

    fig.suptitle('Exact restricted Hartree-Fock calculated orbitals for $%s$ with $N=%d$ and limits=[%d,%d]' % (title_subject, N, limits[IDX_START], limits[IDX_END]))
    axes = []

    # create results table
    for current_energy_level in range(total_energy_levels):

        # reshape eigenvectors for specified
        data = numpy.square(eigenvectors[:,current_energy_level].reshape((N,N,N)).transpose())

        # three rows, two columns, index
        axes.append(fig.add_subplot(3, 2, current_energy_level + 1, projection='3d', proj_type='ortho'))

        # change viewing angles
        axes[-1].view_init(45, 30)

        axes[-1].xaxis.pane.fill = False
        axes[-1].yaxis.pane.fill = False
        axes[-1].zaxis.pane.fill = False

        # set limits
        axes[-1].set_xlim3d(limits[IDX_START], limits[IDX_END])
        axes[-1].set_ylim3d(limits[IDX_START], limits[IDX_END])
        axes[-1].set_zlim3d(limits[IDX_START], limits[IDX_END])

        axes[-1].set_xlabel('$x$')
        axes[-1].set_ylabel('$y$')
        axes[-1].set_zlabel('$z$')

        # set title
        axes[-1].set_title('n = %d' % current_energy_level, y=0.95)

        # create a mask for the data to make the visualization clearer
        mask = data > (data.max() * 0.1)
        idx = numpy.arange(int(numpy.prod(data.shape)))
        x, y, z = numpy.unravel_index(idx, data.shape)
        x, y, z = numpy.meshgrid(coords[IDX_X], coords[IDX_Y], coords[IDX_Z])
        axes[-1].scatter(x, y, z, c=data.flatten(), s=50.0 * mask, edgecolor='face', alpha=0.1, marker='o', cmap='viridis', linewidth=0)

    plot_file_name = results.args.output_file.replace('.xyzp','') + ('.png')
    fig.savefig(plot_file_name)
    # plt.show()

#
# This is the main function
#
def main(cmd_args):

    print('\n** Exact Hartree-Fock simulator **\n')

    # results object
    results = Results()

    ## extract arguments

    print('** Arguments:')

    # input file to process
    input_file = cmd_args.input_file

    # if we have an input file, it will have data and args, load them
    if input_file:
        print('Input %s supplied, reading its data and arguments...' % input_file)
        results.load(input_file)
        args = results.args
        args.input_file = input_file
        eigenvectors = results.eigenvectors
        eigenvalues = results.eigenvalues
    else:
        args = cmd_args

    # print args
    for arg in vars(args):
        print('\t' + arg, getattr(args, arg))

    # output file to save results to
    output_file = args.output_file

    # energy level to converge on
    # always use value provided by command line
    energy_level = cmd_args.energy_level

    # target subject to run sim on
    target_subject = args.target_subject

    # convergence condition percentage
    convergence_percentage = args.convergence_percentage

    # number of eigenvalues to calculate
    total_energy_levels = 6

    # damping factor
    damping_factor = 0.25

    # number of partitions in the solution
    N = args.num_partitions

    # limits of sim
    limits = (-1*args.limit, args.limit)

    # generate coordinates
    coords = generate_coordinates(limits[IDX_START], limits[IDX_END], N)

    # calculate partition size
    h = coords[IDX_X][1]-coords[IDX_X][0]

    ## start program

    print('\n** Program start!\n')

    print('\tPartition size: %f' % h)

    # check if we're simulating something new (no input file specified)
    if not input_file:

        # generate attraction matrix for hydrogen molecule
        attraction_matrix_hydrogen = attraction_matrix_gen(attraction_func_hydrogen, N, coords)

        # generate attraction matrix for helium molecule
        attraction_matrix_helium = attraction_matrix_gen(attraction_func_helium, N, coords)

        # generate laplacian matrix
        laplacian_matrix = second_order_laplacian_3d_sparse_matrix_gen(N)

        # generate kinetic matrix
        kinetic_energy_matrix = (-1.0/(2.0*h**2))*laplacian_matrix

        # create base solution
        eigenvectors = numpy.ndarray(((N**3), total_energy_levels))

        # last eigenvalues
        last_total_energy = 0

        # first iteration
        first_iteration = True

        # iteration counter
        iteration_count = 1

        # total time
        total_time_start = time.time()

        datetime_now = datetime.datetime.now()
        print('\n** Simulation start! %s\n' % datetime_now.strftime(DATETIME_STR_FORMAT))

        # main loop
        while True:

            # iteration time
            iteration_time_start = time.time()

            if first_iteration:
                print('\n** First iteration, zeros used as first guess\n')
                first_iteration = False
            else:
                print('\n** Iteration: %d\n' % iteration_count)

            # Modify eigenvectors to help with convergence
            print('** Modifying eigenvector values with damping factor of %f' % damping_factor)
            eigenvector = eigenvectors[:,energy_level] * damping_factor

            # create integration matrix
            print('** Generating integration matrix')
            # turn our eigenvector into a square matrix and square all of the terms
            orbital_values_squared = numpy.square(eigenvector).reshape((N,N,N)).transpose()

            integration_matrix = integration_matrix_gen(orbital_values_squared, N, coords)

            # create Fock matrix
            if target_subject == 'h2':
                fock_matrix = kinetic_energy_matrix + attraction_matrix_hydrogen + integration_matrix
            elif target_subject == 'he':
                fock_matrix = kinetic_energy_matrix + attraction_matrix_helium + integration_matrix
            else:
                print('Fatal error, exiting.')
                quit()

            # get (total_energy_levels) eigenvectors and eigenvalues and order them from smallest to largest
            print('** Obtaining eigenvalues and eigenvectors...')
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(fock_matrix, k=total_energy_levels, which='SM', maxiter=10000, tol=1e-3)

            # check percentage difference between previous and current eigenvalues
            total_energy = numpy.sum(eigenvalues)

            # calculate total energy
            total_energy_percent_diff = abs(total_energy - last_total_energy)/((total_energy + last_total_energy) / 2)

            print('\n** Total orbital energy %% diff: %.3f%%\n' % (total_energy_percent_diff * 100.0))

            # update last value
            last_total_energy = total_energy

            # update iteration count
            iteration_count = iteration_count + 1

            # append iteration time
            iteration_time = time.time() - iteration_time_start
            results.iteration_times.append(iteration_time)

            datetime_now = datetime.datetime.now()
            print('\n** Iteration end! %s\n' % datetime_now.strftime(DATETIME_STR_FORMAT))
            print('** Iteration time: %.3f seconds**' % iteration_time)
            print('\n** Eigenvalues:\n')
            print(eigenvalues)

            # check if we meet convergence condition
            if abs(total_energy_percent_diff) < (convergence_percentage/100.0):
                break

        # append total time
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        results.total_time = total_time
        datetime_now = datetime.datetime.now()
        print('\n** Simulation end! %s\n' % datetime_now.strftime(DATETIME_STR_FORMAT))
        print('** Total time: %.3f seconds **' % results.total_time)

        # construct file name
        if not output_file:
            output_file = target_subject + '_N%d_l%d.xyzp' % (N, limits[IDX_END])
            args.output_file = output_file

        print('** Saving results to ' + output_file)
        results.eigenvectors = eigenvectors
        results.eigenvalues = eigenvalues
        results.args = args
        results.save(output_file)

    print('\n** Solution summary: **\n')

    print('** Solution took %d iterations to converge with a convergence criteria of %.1f%% in %.3f seconds' % (len(results.iteration_times), results.args.convergence_percentage, results.total_time))
    for iteration, iteration_time in enumerate(results.iteration_times):
        print('\tIteration %i took %.3f seconds' % (iteration, iteration_time))

    # solution results display

    print('\n** Coordinates:\n')
    print(coords)

    print('\n** Eigenvalues:\n')
    print(eigenvalues)

    print('\n** Orbital energies:\n')
    for n in range(len(eigenvalues)):
        print('\tn=%d orbital energy: %f' % (n, eigenvalues[n]))
        eigenvector = eigenvectors[:,n]
        squared_eigenvector_3d = lambda x, y, z : numpy.square(eigenvector).reshape((N,N,N)).transpose()[x, y, z]
        expectation = integrate(squared_eigenvector_3d, coords)
        print('\t\tPsi_%d expectation value: %f' % (n, expectation))

    # print('\n** Total Energies:\n')
    # for n in range(len(eigenvalues)):
    #     eigenvector = eigenvectors[:,n]
    #     print('\tn=%d total energy: %f' % (n, calculate_total_energy(target_subject, eigenvector, coords)))

    # plot data
    print('\n** Plotting data...\n')
    make_plots(results)

#
# This function calculates the total energy of the system
#
def calculate_total_energy(target_subject, orbital_values, coords):

    # extract N
    N = int(numpy.cbrt(len(orbital_values)))

    # extract h
    h = coords[IDX_X][1] - coords[IDX_X][0]

    # turn orbital values into 3d array
    orbital_values_squared = numpy.square(orbital_values.reshape((N,N,N)).transpose())

    # calculate kinetic energy

    # helper lambdas
    # interpret a matrix as a function f(x,y,z)
    matrix_to_func = lambda matrix : lambda x, y, z : matrix[x, y, z]

    # calculate kinetic energy
    diff_result = diff_2(matrix_to_func(orbital_values_squared), coords)
    kinetic_energy = -1.0*(integrate(matrix_to_func(diff_result), coords))

    # calculate nuclear attraction
    if target_subject == 'he':
        attraction_vals = attraction_val_matrix_gen(attraction_func_helium, coords)
    elif target_subject == 'h2':
        attraction_vals = attraction_val_matrix_gen(attraction_func_hydrogen, coords)
    else:
        print('Fatal error, exiting.')
        quit()

    nuclear_attraction = 2*integrate(matrix_to_func(orbital_values_squared*attraction_vals), coords)

    # calculate Coulomb/exchange contribution
    two_electron_integration = two_electron_integration_calc(orbital_values_squared, coords)

    return kinetic_energy + nuclear_attraction + two_electron_integration

#
# This functions returns whether or not the matrix is symmetric
#
def is_symmetric(A, tol=1e-8):
    return scipy.sparse.linalg.norm(A-A.T, scipy.Inf) < tol;

#
# This functions returns whether or not the matrix is symmetric
#
def is_hermitian(A, tol=1e-8):
    return scipy.sparse.linalg.norm(A-A.H, scipy.Inf) < tol;

#
# This function takes in a matrix index (row or column) and returns the
# associated coordinate indices as a tuple.
#
@functools.lru_cache
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
@functools.lru_cache
def coordinate_index_to_coordinates(coord_indices, coords):

    return (coords[IDX_X][coord_indices[IDX_X]], coords[IDX_Y][coord_indices[IDX_Y]], coords[IDX_Z][coord_indices[IDX_Z]])

#
# Generate the coordinates for the solution space, with the assumption that it
# is cubic.
#
def generate_coordinates(minimum, maximum, N):

    x = tuple(numpy.linspace(minimum, maximum, N))
    y = tuple(numpy.linspace(minimum, maximum, N))
    z = tuple(numpy.linspace(minimum, maximum, N))

    return (x, y, z)

#
# This function returns the nuclear attraction for an electron in the Helium
# element simulation. A small number tiny_number is provided to prevent divide
# by zero scenarios.
#
def attraction_func_helium(coords, h):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    tiny_number = TINY_NUMBER

    denominator = math.sqrt(x**2 + y**2 + z**2)

    return -((2.0/(tiny_number + denominator)))

#
# This function returns the nuclear attraction for an electron in the Hydrogen
# molecule simulation. A small number tiny_number is provided to prevent divide
# by zero scenarios.
#
def attraction_func_hydrogen(coords, h):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    tiny_number = TINY_NUMBER

    denominator_1 = math.sqrt(((H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + z**2)
    denominator_2 = math.sqrt(((-H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + z**2)

    return -((1.0/(tiny_number + denominator_1)) + (1.0/(tiny_number + denominator_2)))

#
# This functions calculates the repulsion between two electrons. A small number
# TINY_NUMBER is provided to prevent divide by zero scenarios.
#
@functools.lru_cache(maxsize=8192)
def repulsion_func(coords_1, coords_2, h):

    x1 = coords_1[IDX_X]
    y1 = coords_1[IDX_Y]
    z1 = coords_1[IDX_Z]

    x2 = coords_2[IDX_X]
    y2 = coords_2[IDX_Y]
    z2 = coords_2[IDX_Z]

    tiny_number = TINY_NUMBER

    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return ((1.0/(tiny_number + denominator)))

# This function generates an N*N*N matrix A with the specified attraction
# function. This matrix contains values that evaluates the attraction
# function at the specified coordinates. i.e, A[x,y,z] = attraction_func
# (x,y,z) Note that this is different from the matrix generated by
# attraction_matrix_gen, which is used to solve a linear system of equations.
def attraction_val_matrix_gen(attraction_func, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # generate a solution space
    solution_space = numpy.empty((N,N,N))

    for xi in range(N):
        for yi in range(N):
            for zi in range(N):

                x = coords[IDX_X][xi]
                y = coords[IDX_Y][yi]
                z = coords[IDX_Z][zi]

                solution_space[xi,yi,zi] = attraction_func((x, y, z), h)

    return solution_space

#
# This function generates an N*N*N matrix A with the inner result of the
# integration term (combined Coulomb/exchange)
#
def inner_integration_val_matrix_gen(orbital_values_squared, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # generate a solution space
    solution_space = numpy.empty((N,N,N))

    for xi in range(N):
        for yi in range(N):
            for zi in range(N):

                x = coords[IDX_X][xi]
                y = coords[IDX_Y][yi]
                z = coords[IDX_Z][zi]

                solution_space[xi,yi,zi] = integration_term_func_new(orbital_values_squared, (x, y, z), coords)

    # update progress bar


    return solution_space

#
# This function calculates the Coulomb/exchange two electron integral
# expectation value used in the total energy calculation.
#
def two_electron_integration_calc(orbital_values_squared, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # calculate inner integral matrix
    inner_integration_val_matrix = inner_integration_val_matrix_gen(orbital_values_squared, coords)

    # running sum
    sum = 0

    # calculate the integration over the solution space of the specified psi squared
    for xi in range(N):
        for yi in range(N):
            for zi in range(N):
                # first integration weight is 0
                if xi == 0 or yi == 0 or zi == 0:
                    w = 0
                else:
                    w = h
                sum = sum + w*(orbital_values_squared[xi, yi, zi]*inner_integration_val_matrix[xi, yi, zi])

    return sum

#
# This function generates an attraction matrix with the specified attraction
# function and coordinates
#
def attraction_matrix_gen(attraction_func, N, coords):

    # extract h
    h = coords[IDX_X][1] - coords[IDX_X][0]

    # use scipy sparse matrix generation
    # create the diagonal 
    diagonal = [attraction_func(coordinate_index_to_coordinates(matrix_index_to_coordinate_indices(i, N), coords), h) for i in range(N**3)]

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
# This is a generic numerical second-order central 3D differentiation (the same
# used in the matrix solution)
#
def diff_2(function, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # generate a solution space
    solution_space = numpy.empty((N,N,N))

    for xi in range(N):
        for yi in range(N):
            for zi in range(N):

                # central requires values from the past, if required, use 0
                if xi == 0:
                    x_past = 0
                else:
                    x_past = function(xi-1,yi,zi)

                if yi == 0:
                    y_past = 0
                else:
                    y_past = function(xi,yi-1,zi)

                if zi == 0:
                    z_past = 0
                else:
                    z_past = function(xi,yi,zi-1)

                # we'll always have present values
                present = function(xi,yi,zi)

                # central requires values from the future, if required, use 0
                if xi == (N-1):
                    x_future = 0
                else:
                    x_future = function(xi+1,yi,zi)

                if yi == (N-1):
                    y_future = 0
                else:
                    y_future = function(xi,yi+1,zi)

                if zi == (N-1):
                    z_future = 0
                else:
                    z_future = function(xi,yi,zi+1)

                # differentiate
                solution_space[xi,yi,zi] = ((x_future + y_future + z_future - (6*present) + x_past + y_past + z_past)/(h**2))

    return solution_space

#
# This is a generic numerical integration over the cubic 3D space covered by the
# solution space.
#
def integrate(function, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # running sum
    sum = 0

    # calculate the integration over the solution space
    for xi in range(N):
        for yi in range(N):
            for zi in range(N):

                # first integration weight is 0
                if xi == 0 or yi == 0 or zi == 0:
                    w = 0
                else:
                    w = h
                row_index = xi + yi*N + zi*N*N
                sum = sum + w*function(xi, yi, zi)

    # return result
    return sum

#
# This function evaluates the integration function used by both the Helium
# element and the Hydrogen molecule.
#
def integration_term_func(orbital_values_squared, all_coords, coords_1):

    # get partition size
    h = all_coords[IDX_X][1] - all_coords[IDX_X][0]
    # get number of partitions
    N = len(all_coords[IDX_X])

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
                sum = sum + w*orbital_values_squared[xi, yi, zi]*repulsion_func(coords_1, coords_2, h)

    return sum

#
# This function evaluates the integration function used by both the Helium
# element and the Hydrogen molecule. This is a modified version from
# integration_term_func to support the total energy calculation. It mostly
# changes the way things are passed in.
#
def integration_term_func_new(orbital_values_squared, coords_1, all_coords):

    # get partition size
    h = all_coords[IDX_X][1] - all_coords[IDX_X][0]
    # get number of partitions
    N = len(all_coords[IDX_X])

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
                sum = sum + w*(orbital_values_squared[xi, yi, zi]*repulsion_func(coords_1, coords_2, h))

    return sum


# This function generates the the integration matrix
#
def integration_matrix_gen(orbital_values_squared, N, coords):


    # extract h from coordinates
    h = coords[IDX_X][1] - coords[IDX_X][0]

    # generate coordinates
    coordinates = [coordinate_index_to_coordinates(matrix_index_to_coordinate_indices(i, N), coords) for i in range(N**3)]

    # this takes an extremely long time!!!
    if ENABLE_MP:
        # multiprocessing filling of the diagonal
        with multiprocessing.Pool(processes = multiprocessing.cpu_count()-1, maxtasksperchild=1000) as pool:
            func = functools.partial(integration_term_func, orbital_values_squared, coords)
            # diagonal = pool.map(func, coordinates)
            diagonal = list(tqdm.tqdm(pool.imap(func, coordinates), total=(N**3), ascii=True))
    else:
        # create the diagonal (no MP)
        progress_bar = progress.bar.ShadyBar('\tGenerating diagonal for matrix...', max=(N**3), suffix='%(index)d/%(max)d - %(percent).1f%%')
        diagonal = numpy.ndarray(N**3)
        for i in range(N**3):
            diagonal[i] = integration_term_func(orbital_values_squared, coords, coordinates[i])

    # add a new line after we're done progress
    print()

    # now generate the matrix with the desired diagonal
    matrix = scipy.sparse.spdiags(data=diagonal, diags=0, m=N**3, n=N**3)

    return matrix.tocoo()

if __name__ == '__main__':
    # the following sets up the argument parser for the program
    parser = argparse.ArgumentParser(description='Exact Hartree-Fock simulator')

    # arguments for the program
    parser.add_argument('-i', type=str, dest='input_file', action='store',
        help='input file (*.xyzp) to plot')

    parser.add_argument('-o', type=str, dest='output_file', action='store',
        help='path and name of the file containing the results of the current run')

    parser.add_argument('-e', type=int, default=0, dest='energy_level', action='store', choices=[0, 1, 2, 3, 4, 5],
        help='energy level to generate and/or plot')

    parser.add_argument('-t', type=str, default='h2', dest='target_subject', action='store', choices=['h2', 'he'],
        help='target subject to run exact HF sim on')

    parser.add_argument('-p', type=int, default=14, dest='num_partitions', action='store',
        help='number of partitions to discretize the simulation')

    parser.add_argument('-l', type=float, default=5, dest='limit', action='store',
        help='the x,y,z max limit, forming a cubic solution space')

    parser.add_argument('-c', type=float, default=1.0, dest='convergence_percentage', action='store',
        help='percent change threshold for convergence')

    args = parser.parse_args()

    main(args)