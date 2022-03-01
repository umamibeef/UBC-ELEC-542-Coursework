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
import scipy.integrate
import sympy
import sympy.vector
import math
import functools
import itertools
import time
import pickle
import datetime # for timestamping
import multiprocessing # for multiprocessing (MP) of matrix generation
import tqdm # progress bar for MP
import sys

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
TINY_NUMBER = 0.001
IDX_X = 0
IDX_Y = 1
IDX_Z = 2
IDX_START = 0
IDX_END = 1
DATETIME_STR_FORMAT = '[%Y/%m/%d-%H:%M:%S]'
ENABLE_MP = True # multiprocessing

#
# This is the main function
#
def main(cmd_args):

    precalculate_integrals()

#
# Helium STO-1G basis function (centered around origin)
#
sto_1g_helium_func = lambda z, y, x: 0.5881*sympy.exp(-0.7739*(sympy.sqrt(x**2+y**2+z**2))**2)

#
# Hydrogen STO-1G basis function (centered around Hydrogen nucleus 0)
#
sto_1g_hydrogen_0_func = lambda z, y, x: 0.3696*sympy.exp(-0.4166*(sympy.sqrt((x - (-H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2+y**2+z**2))**2)

#
# Hydrogen STO-1G basis function (centered around Hydrogen nucleus 1)
#
sto_1g_hydrogen_1_func = lambda z, y, x: 0.3696*sympy.exp(-0.4166*(sympy.sqrt((x - (H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2+y**2+z**2))**2)

#
# Pre-calculate integrals
#
def precalculate_integrals():

    # sympy 3D x,y,z, coordinate system
    R = sympy.vector.CoordSys3D('R')
    x, y, z = sympy.symbols('x y z')

    func = sto_1g_helium_func(R.z, R.y, R.x)*sympy.vector.Laplacian(sto_1g_helium_func(R.z, R.y, R.x)).doit()
    func_num = sympy.lambdify([R.z, R.y, R.x], func, 'scipy')
    print(scipy.integrate.tplquad(func_num, -1000, 1000, lambda x: -1000, lambda x: 1000, lambda x, y: -1000, lambda x, y: 1000))

    # cartesian product for all combinations
    combinations = itertools.product([0,1],repeat=2)
    # unique combinations
    combinations = itertools.combinations_with_replacement([0,1],2)
    for combination in combinations:
        print(combination)
    # check for combination
    set_0 = [1,0]
    set_1 = [0,1]
    # sort and compare
    print(sorted(set_0) == sorted(set_1))
    # hash for lookup
    # take care of negative hash
    mask = (1<<sys.hash_info.width) - 1
    print('%X' % (hash(tuple(sorted(set_0))) & mask))

    # print(sympy.integrate(sympy.exp(-R.x**2 - R.y**2), (R.x, -sympy.oo, sympy.oo), (R.y, -sympy.oo, sympy.oo)).doit())
    # sympy.integrate(sto_1g_helium_func(R.x, R.y, R.z), (R.x, -sympy.oo, sympy.oo), (R.y, -sympy.oo, sympy.oo), (R.z, -sympy.oo, sympy.oo))

    # one-electron integral

    he_overlap_ints = {}
    he_kinetic_energy_ints = {}
    he_nuclear_attraction_ints = {}

    h_overlap_ints = {}
    h_kinetic_energy_ints = {}
    h_nuclear_attraction_ints = {}

    # two-electron integrals

    # coulomb repulsion and exchange integrals

    pass

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

    parser.add_argument('-d', type=float, default=0.1, dest='damping_factor', action='store',
        help='damping factor to apply to orbital results between iterations')

    args = parser.parse_args()

    main(args)