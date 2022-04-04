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
AU_DISTANCE=5.29e-11; # Atomic unit of distance= Bohr radius (m)
H2_BOND_LENGTH_ATOMIC_UNITS = 0.74e-10/AU_DISTANCE; # Bond length of Hydrogen atom in atomic unit
TINY_NUMBER = 1e-3
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
        # keep track of total energies and eigenvalues
        self.historical_total_orbital_energies = []
        self.historical_eigenvalues = []

    def load(self, input_file):
        with open(input_file, 'rb') as file:
            tmp_dict = pickle.load(file)         

        self.__dict__.update(tmp_dict) 

    def save(self, output_file):
        with open(output_file, 'wb') as file:
            pickle.dump(self.__dict__, file, 2)

    def display_data(self):
        for key in self.__dict__:
            console_print(key)
            console_print(self.__dict__[key])

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
        data = numpy.square(numpy.array(eigenvectors[:,current_energy_level]).reshape((N,N,N)).transpose())

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
# Console formatter
#
def console_print(string=''):

    # get str representation
    if not isinstance(string, str):
        string = str(string)

    datetime_now = datetime.datetime.now()
    print(datetime_now.strftime(DATETIME_STR_FORMAT) + ' ' + string)

#
# This is the main function
#
def main(cmd_args):

    console_print(' ** Exact Hartree-Fock simulator **')

    # results object
    results = Results()

    ## extract arguments

    console_print(' ** Arguments:')

    # input file to process
    input_file = cmd_args.input_file

    # if we have an input file, it will have data and args, load them
    if input_file:
        console_print('Input %s supplied, reading its data and arguments...' % input_file)
        results.load(input_file)
        args = results.args
        args.input_file = input_file
        eigenvectors = results.eigenvectors
        eigenvalues = results.eigenvalues
    else:
        args = cmd_args

    # print args
    for arg in vars(args):
        console_print('\t' + arg + ' ' + str(getattr(args, arg)))

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
    damping_factor = args.damping_factor

    # number of partitions in the solution
    N = args.num_partitions

    # limits of sim
    limits = (-1*args.limit, args.limit)

    # generate coordinates
    coords = generate_coordinates(limits[IDX_START], limits[IDX_END], N)

    # calculate partition size
    h = coords[IDX_X][1]-coords[IDX_X][0]

    ## start program

    console_print(' ** Program start!')

    console_print('\tPartition size: %f' % h)

    # check if we're simulating something new (no input file specified)
    if not input_file:

        if target_subject == 'h2':
            # generate attraction matrix for hydrogen molecule
            attraction_matrix = attraction_matrix_gen(attraction_func_hydrogen, coords)
        elif target_subject == 'he':
            # generate attraction matrix for helium molecule
            attraction_matrix = attraction_matrix_gen(attraction_func_helium, coords)
        else:
            console_print('Fatal error, exiting.')
            quit()

        # generate laplacian matrix
        laplacian_matrix = second_order_laplacian_3d_sparse_matrix_gen(N)

        # generate kinetic matrix
        kinetic_energy_matrix = (laplacian_matrix/(2.0*h**2))

        # create base solution
        initial_fock_matrix = -kinetic_energy_matrix - attraction_matrix
        eigenvalues, eigenvectors = numpy.linalg.eigh(initial_fock_matrix)

        # last total energy
        last_total_energy = 0

        # first iteration
        first_iteration = True

        # iteration counter
        iteration_count = 1

        # total time
        total_time_start = time.time()

        console_print(' ** Simulation start! ')

        # main loop
        while True:

            # iteration time
            iteration_time_start = time.time()

            if first_iteration:
                console_print(' ** First iteration, zeros used as first guess')
                first_iteration = False
            else:
                console_print(' ** Iteration: %d' % iteration_count)

            # Modify eigenvectors to help with convergence
            console_print(' ** Modifying eigenvector values with damping factor of %f' % damping_factor)
            eigenvector = numpy.array(eigenvectors[:,energy_level]) * damping_factor

            # create integration matrix
            console_print(' ** Generating integration matrix')
            # turn our eigenvector into a square matrix and square all of the terms
            # orbital_values_squared = numpy.square(eigenvector).reshape((N,N,N)).transpose()
            orbital_values = eigenvector.reshape(N,N,N) #.transpose()

            # create coulomb repulsion matrix
            repulsion_matrix = repulsion_matrix_gen(orbital_values, coords)

            # create exchange matrix
            exchange_matrix = exchange_matrix_gen(orbital_values, coords)

            # create Fock matrix
            fock_matrix = -kinetic_energy_matrix - attraction_matrix + 2*repulsion_matrix - exchange_matrix

            # get (total_energy_levels) eigenvectors and eigenvalues and order them from smallest to largest
            console_print(' ** Obtaining eigenvalues and eigenvectors...')
            eigenvalues, eigenvectors = numpy.linalg.eigh(fock_matrix)

            # remove extra solutions by trimming down to the amount we desire
            eigenvectors = eigenvectors[:,:total_energy_levels]
            eigenvalues = eigenvalues[:total_energy_levels]

            # calculate total energy
            energies = []
            # sum for total number of electrons
            for i in range(2):
                # calculate total energy = psi'*(H-attraction)*psi + 0.5*psi'*(2*repulsion - exchange)*psi
                psi_prime = numpy.matrix(eigenvectors[:,i]).getH()
                psi = eigenvectors[:,i]
                energies += [psi_prime*(-kinetic_energy_matrix - attraction_matrix)*psi + 0.5*psi_prime*(2*repulsion_matrix - exchange_matrix)*psi]
            total_energy = sum(energies)

            # record history
            results.historical_eigenvalues.append(eigenvalues)
            results.historical_total_orbital_energies.append(total_energy)

            # calculate total energy percent difference
            total_energy_percent_diff = abs((total_energy - last_total_energy)/((total_energy + last_total_energy) / 2))

            console_print(' ** Total energy: %.3f' % (total_energy))
            console_print(' ** Energy %% diff: %.3f%%' % (total_energy_percent_diff * 100.0))

            # update last value
            last_total_energy = total_energy

            # update iteration count
            iteration_count = iteration_count + 1

            # append iteration time
            iteration_time = time.time() - iteration_time_start
            results.iteration_times.append(iteration_time)

            console_print(' ** Iteration end! Iteration time: %.3f seconds**' % iteration_time)
            console_print(' ** Eigenvalues:')
            console_print(eigenvalues)

            # check if we meet convergence condition
            if abs(total_energy_percent_diff) < (convergence_percentage/100.0):
                break

        # append total time
        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        results.total_time = total_time
        console_print(' ** Simulation end! ')
        console_print(' ** Total time: %.3f seconds **' % results.total_time)

        # construct file name
        if not output_file:
            output_file = target_subject + '_N%d_l%d.xyzp' % (N, limits[IDX_END])
            args.output_file = output_file

        console_print(' ** Saving results to ' + output_file)
        results.eigenvectors = eigenvectors[:,:total_energy_levels]
        results.eigenvalues = eigenvalues[:total_energy_levels]
        results.args = args
        results.save(output_file)

    console_print(' ** Solution summary: **')

    console_print(' ** Solution took %d iterations to converge with a convergence criteria of %.1f%% in %.3f seconds' % (len(results.iteration_times), results.args.convergence_percentage, results.total_time))
    for iteration, iteration_time in enumerate(results.iteration_times):
        console_print('\tIteration %i took %.3f seconds' % (iteration, iteration_time))

    console_print(' ** Orbital energies:')
    for n in range(len(eigenvalues)):
        console_print('\tn=%d orbital energy: %f' % (n, eigenvalues[n]))

    # plot data
    # console_print(' ** Plotting data...')
    # make_plots(results)

#
# This function calculates the total energy of the system
#
def calculate_total_energy(target_subject, orbital_values, coords):

    # extract N
    N = int(numpy.cbrt(len(orbital_values)))

    # extract h
    h = coords[IDX_X][1] - coords[IDX_X][0]

    # turn orbital values into 3d array
    orbital_values_squared = numpy.square(orbital_values.reshape((N,N,N))) #.transpose())

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
        console_print('Fatal error, exiting.')
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
def coordinate_indices_to_coordinates(coord_indices, coords):

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
def attraction_func_helium(coords):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    denominator = math.sqrt(x**2 + y**2 + (z+0.5)**2)

    # check if we have infinite solutions
    if not denominator:
        denominator = TINY_NUMBER

    return ((2.0/(denominator)))

#
# This function returns the nuclear attraction for an electron in the Hydrogen
# molecule simulation. A small number tiny_number is provided to prevent divide
# by zero scenarios.
#
def attraction_func_hydrogen(coords):

    x = coords[IDX_X]
    y = coords[IDX_Y]
    z = coords[IDX_Z]

    denominator_1 = math.sqrt(((H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + (z+0.5)**2)
    denominator_2 = math.sqrt(((-H2_BOND_LENGTH_ATOMIC_UNITS/2) - x)**2 + y**2 + (z+0.5)**2)

    # check if we have infinite solutions
    if not denominator_1:
        denominator_1 = TINY_NUMBER
    if not denominator_2:
        denominator_2 = TINY_NUMBER

    return ((1.0/(denominator_1)) + (1.0/(denominator_2)))

#
# This functions calculates the repulsion between two electrons. A small number
# TINY_NUMBER is provided to prevent divide by zero scenarios.
#
@functools.lru_cache(maxsize=8192)
def repulsion_func(coords_1, coords_2):

    x1 = coords_1[IDX_X]
    y1 = coords_1[IDX_Y]
    z1 = coords_1[IDX_Z]

    x2 = coords_2[IDX_X]
    y2 = coords_2[IDX_Y]
    z2 = coords_2[IDX_Z]

    denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    # check if we have infinite solutions
    if not denominator:
        denominator = TINY_NUMBER

    return ((1.0/(denominator)))

#
# This function generates an attraction matrix with the specified attraction
# function and coordinates
#
def attraction_matrix_gen(attraction_func, coords):

    # get partition size
    h = coords[IDX_X][1] - coords[IDX_X][0]
    # get number of partitions
    N = len(coords[IDX_X])

    # print('h: %f' % h)

    # use scipy sparse matrix generation
    # create the diagonal 
    diagonal = [attraction_func(coordinate_indices_to_coordinates(matrix_index_to_coordinate_indices(i, N), coords)) for i in range(N**3)]

    # now generate the matrix with the desired diagonal
    matrix = numpy.diag(diagonal)

    return matrix

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
    for i in range(N**3):
        coords_1 = matrix_index_to_coordinate_indices(i, N)
        sum += function(coords_1)*(h**3)

    # return result
    return sum

#
# This function evaluates the integrand of the repulsion matrix diagonals
#
def repulsion_matrix_integrand_func(orbital_values, coords_1, coords_2, coords):

    x2 = coords_2[IDX_X]
    y2 = coords_2[IDX_Y]
    z2 = coords_2[IDX_Z]

    real_coords_1 = coordinate_indices_to_coordinates(coords_1, coords)
    real_coords_2 = coordinate_indices_to_coordinates(coords_2, coords)

    # OK, this was a headspin!!! Orbital values are indexed by normal indices,
    # but we need to make sure we're using actual coordinates for the repulsion function!

    return (orbital_values[x2, y2, z2]**2)*repulsion_func(real_coords_1, real_coords_2)

#
# This function evaluates the Coulomb term 
#
def repulsion_matrix_gen(orbital_values, coords):

    # extract N, get number of partitions
    N = len(coords[IDX_X])

    # generate the diagonal
    diagonal = []
    for i in range(N**3):
        coords_1 = matrix_index_to_coordinate_indices(i, N)
        integrand = lambda coords_2 : repulsion_matrix_integrand_func(orbital_values, coords_1, coords_2, coords)
        diagonal += [integrate(integrand, coords)]

    # now generate the matrix with the desired diagonal
    matrix = numpy.diag(diagonal)

    return matrix

#
# This funciton evaluates the integrand of the exchange matrix diagonals
#
def exchange_matrix_integrand_func(orbital_values, coords_1, coords_2, coords):

    x1 = coords_1[IDX_X]
    y1 = coords_1[IDX_Y]
    z1 = coords_1[IDX_Z]

    x2 = coords_2[IDX_X]
    y2 = coords_2[IDX_Y]
    z2 = coords_2[IDX_Z]

    real_coords_1 = coordinate_indices_to_coordinates(coords_1, coords)
    real_coords_2 = coordinate_indices_to_coordinates(coords_2, coords)

    # OK, this was a headspin!!! Orbital values are indexed by normal indices,
    # but we need to make sure we're using actual coordinates for the repulsion function!

    return orbital_values[x1, y1, z1]*orbital_values[x2, y2, z2]*repulsion_func(real_coords_1, real_coords_2)

#
# This funciton evaluates the Exchange term
#
def exchange_matrix_gen(orbital_values, coords):

    h = coords[IDX_X][1] - coords[IDX_X][0]
    # extract N, get number of partitions
    N = len(coords[IDX_X])

    # diagonal = numpy.zeros(N**3)
    # for i in range(N**3):
    #     coords_1_idx = matrix_index_to_coordinate_indices(i, N)
    #     coords_1_vals = coordinate_indices_to_coordinates(coords_1_idx, coords)
    #     for j in range(N**3):
    #         coords_2_idx = matrix_index_to_coordinate_indices(j, N)
    #         coords_2_vals = coordinate_indices_to_coordinates(coords_2_idx, coords)

    #         x1 = coords_1_idx[IDX_X]
    #         y1 = coords_1_idx[IDX_Y]
    #         z1 = coords_1_idx[IDX_Z]
    #         x2 = coords_2_idx[IDX_X]
    #         y2 = coords_2_idx[IDX_Y]
    #         z2 = coords_2_idx[IDX_Z]

    #         diagonal[i] += orbital_values[x1, y1, z1]*orbital_values[x2, y2, z2]*repulsion_func(coords_1_vals, coords_2_vals)*(h**3)

    # exchange_matrix = numpy.diag(diagonal)

    # exchange_matrix = numpy.zeros((N**3,N**3))
    # for i in range(N**3):
    #     coords_1_idx = matrix_index_to_coordinate_indices(i, N)
    #     coords_1_vals = coordinate_indices_to_coordinates(coords_1_idx, coords)
    #     for j in range(i+1):
    #         coords_2_idx = matrix_index_to_coordinate_indices(j, N)
    #         coords_2_vals = coordinate_indices_to_coordinates(coords_2_idx, coords)

    #         x1 = coords_1_idx[IDX_X]
    #         y1 = coords_1_idx[IDX_Y]
    #         z1 = coords_1_idx[IDX_Z]

    #         x2 = coords_2_idx[IDX_X]
    #         y2 = coords_2_idx[IDX_Y]
    #         z2 = coords_2_idx[IDX_Z]

    #         exchange_matrix[i,j] = orbital_values[x1, y1, z1]*orbital_values[x2, y2, z2]*repulsion_func(coords_1_vals, coords_2_vals)
    #         exchange_matrix[j,i] = exchange_matrix[i, j]
    # exchange_matrix *= h**3

    # print(exchange_matrix)

    # generate the diagonal
    diagonal = []
    for i in range(N**3):
        coords_1 = matrix_index_to_coordinate_indices(i, N)
        integrand = lambda coords_2 : exchange_matrix_integrand_func(orbital_values, coords_1, coords_2, coords)
        diagonal += [integrate(integrand, coords)]

    # now generate the matrix with the desired diagonal
    exchange_matrix = numpy.diag(diagonal)

    return exchange_matrix

    # return exchange_matrix

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

    parser.add_argument('-t', type=str, default='he', dest='target_subject', action='store', choices=['h2', 'he'],
        help='target subject to run exact HF sim on')

    parser.add_argument('-p', type=int, default=11, dest='num_partitions', action='store',
        help='number of partitions to discretize the simulation')

    parser.add_argument('-l', type=float, default=4, dest='limit', action='store',
        help='the x,y,z max limit, forming a cubic solution space')

    parser.add_argument('-c', type=float, default=1.0, dest='convergence_percentage', action='store',
        help='percent change threshold for convergence')

    parser.add_argument('-d', type=float, default=1.0, dest='damping_factor', action='store',
        help='damping factor to apply to orbital results between iterations')

    args = parser.parse_args()

    main(args)