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
import numpy as np
import math

# Matplotlib export settings
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
TINY_NUMBER = 1e-9
IDX_X = 0
IDX_Y = 1
IDX_Z = 2

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

    IDX_X = np.linspace(minimum, maximum, N)
    IDX_Y = np.linspace(minimum, maximum, N)
    IDX_Z = np.linspace(minimum, maximum, N)

    return (IDX_X, IDX_Y, IDX_Z)

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

    return -((1.0/(TINY_NUMBER + denominator)))

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
# This function generates an attraction matrix with the specified attraction
# function and coordinates
#
def attraction_matrix_gen(attraction_func, N, coords):

    # create an empty matrix with the correct dimensions
    matrix = np.zeros((N**3,N**3))
    # fill in the diagonal with the evaluated attraction function
    for i in range(N**3):
        for j in range(N**3):
            # diagonal
            if (i == j):
                coordinate_indices = matrix_index_to_coordinate_indices(i, N)
                matrix[i][j] = attraction_func(coordinate_index_to_coordinates(coordinate_indices, coords))

    return matrix

def second_order_laplacian_3d_sparse_matrix_gen(dimensions):
    pass

def make_plots(results):
    pass

def main():

    # number of partitions in the solution
    N = 15

    # generate coordinates
    coords = generate_coordinates(-5, 5, N)

    # generate attraction matrix for hydrogen molecule
    attraction_matrix_hydrogen = attraction_matrix_gen(attraction_func_hydrogen, N, coords)

    # generate attraction matrix for helium molecule
    attraction_matrix_helium = attraction_matrix_gen(attraction_func_helium, N, coords)

    # TODO: Implement laplacian matrix generator
    # TODO: Implement numeric integration function
    # TODO: Implement integration term matrix generator

if __name__ == "__main__":
    main()
