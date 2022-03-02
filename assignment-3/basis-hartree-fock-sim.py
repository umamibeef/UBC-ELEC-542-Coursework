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
import json

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
HE_NUM_BASIS_FUNCTIONS = 2
H2_NUM_BASIS_FUNCTIONS = 4
HE_INTEGRALS_FILENAME = 'he_integrals.json'
H2_INTEGRALS_FILENAME = 'h2_integrals.json'

#
# Console formatter
#
def console_print(string='', end='\n'):

    # get str representation
    if not isinstance(string, str):
        string = str(string)

    datetime_now = datetime.datetime.now()
    print(datetime_now.strftime(DATETIME_STR_FORMAT) + ' ' + string, end=end)

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
# Calculate the overlap integrals given the function lookup table and the desired combinations
#
def calculate_overlap_integrals(func_lut, combinations):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    overlap_ints = {}

    for combination in combinations:
        console_print('Calculating overlap integral (%d,%d)...' % (combination[0], combination[1]))
        # symbolic version of the integrand
        overlap_intgd_sym = func_lut[combination[0]](z, y, x)*func_lut[combination[1]](z, y, x)
        # numerical version of the integrand
        overlap_intgd_num = sympy.lambdify([z, y, x], overlap_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        overlap_int_val = scipy.integrate.tplquad(overlap_intgd_num, -scipy.inf, scipy.inf, lambda x: -scipy.inf, lambda x: scipy.inf, lambda x, y: -scipy.inf, lambda x, y: scipy.inf)[0]
        # add integration results to dictionary
        overlap_ints[combination] = overlap_int_val

    return overlap_ints

#
# Calculate the kinetic energy integrals given the function lookup table and the desired combinations
#
def calculate_kinetic_energy_integrals(func_lut, combinations):

    # sympy vector x,y,z coordinate system
    R = sympy.vector.CoordSys3D('R')

    kinetic_energy_ints = {}

    for combination in combinations:
        console_print('Calculating kinetic energy integral (%d,%d)...' % (combination[0], combination[1]))
        # symbolic version of the integrand
        kinetic_energy_intgd_sym = func_lut[combination[0]](R.z, R.y, R.x)*(-1/2)*sympy.vector.Laplacian(h_func_lut[combination[1]](R.z, R.y, R.x)).doit()
        # numerical version of the integrand
        kinetic_energy_intgd_num = sympy.lambdify([R.z, R.y, R.x], kinetic_energy_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        kinetic_energy_int_val = scipy.integrate.tplquad(kinetic_energy_intgd_num, -scipy.inf, scipy.inf, lambda x: -scipy.inf, lambda x: scipy.inf, lambda x, y: -scipy.inf, lambda x, y: scipy.inf)[0]
        # add integration results to dictionary
        kinetic_energy_ints[combination] = kinetic_energy_int_val

    return kinetic_energy_ints

#
# Calculate the nuclear attraction integrals given the function lookup table and the desired combinations
#
def calculate_nuclear_attraction_integrals(func_lut, combinations, subject):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    nuclear_attraction_ints = {}

    for combination in combinations:
        console_print('Calculating nuclear attraction integral (%d,%d)...' % (combination[0], combination[1]))
        # symbolic version of the integrand
        if subject == 'he':
            nuclear_attraction_intgd_sym = func_lut[combination[0]](z, y, x) * (-2/sympy.sqrt(x**2 + y**2 + z**2)) * func_lut[combination[1]](z, y, x)
        elif subject == 'h2':
            nuclear_attraction_intgd_sym = func_lut[combination[0]](z, y, x) * (-1/sympy.sqrt((x - (-H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) - (-1/sympy.sqrt((x - (H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) * func_lut[combination[1]](z, y, x)
        # numerical version of the integrand
        nuclear_attraction_intgd_num = sympy.lambdify([z, y, x], nuclear_attraction_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        nuclear_attraction_int_val = scipy.integrate.tplquad(nuclear_attraction_intgd_num, -scipy.inf, scipy.inf, lambda x: -scipy.inf, lambda x: scipy.inf, lambda x, y: -scipy.inf, lambda x, y: scipy.inf)[0]
        # add integration results to dictionary
        nuclear_attraction_ints[combination] = nuclear_attraction_int_val

    return nuclear_attraction_ints

#
# Calculate the Coulomb repulsion and exchange integrals given the function lookup table and the desired combinations
#
def calculate_coulomb_repulsion_and_exchange_integrals(func_lut, combinations):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')
    u, v, w = sympy.symbols('u v w')

    coulomb_repulsion_and_exchange_ints = {}

    for combination in combinations:
        console_print('Calculating Coulomb repulsion and exchange integrals (%d%d|%d%d)...' % (combination[0], combination[1], combination[2], combination[3]))
        # symbolic version of the integrand
        coulomb_repulsion_and_exchange_intgd_sym = func_lut[combination[0]](z, y, x) * func_lut[combination[1]](z, y, x) * (1/sympy.sqrt((u-x)**2 + (v-y)**2 + (w-z)**2)) * func_lut[combination[2]](w, v, u) * func_lut[combination[3]](w, v, u)
        # numerical version of the integrand
        coulomb_repulsion_and_exchange_intgd_num = sympy.lambdify([z, y, x, w, v, u], coulomb_repulsion_and_exchange_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        coulomb_repulsion_and_exchange_int_val = scipy.integrate.nquad(coulomb_repulsion_and_exchange_intgd_num, -scipy.inf, scipy.inf, -scipy.inf, scipy.inf, -scipy.inf, scipy.inf, -scipy.inf, scipy.inf, -scipy.inf, scipy.inf, -scipy.inf, scipy.inf)[0]
        # add integration results to dictionary
        coulomb_repulsion_and_exchange_ints[combination] = coulomb_repulsion_and_exchange_int_val

    return coulomb_repulsion_and_exchange_ints

#
# Generate 
#
def get_one_electron_combinations(num_basis_functions):

    combinations = list(itertools.combinations_with_replacement(list(range(num_basis_functions)),2))
    console_print('  One-Electron Integral Combinations (total=%d):' % len(combinations))
    for combination in combinations:
        console_print('    (%d, %d)' % (combination[0], combination[1]))

    return combinations

#
# Generate two electron integral combinations
#
def get_two_electron_combinations(num_basis_functions):
    # generate combinations
    # G_{row/v,col/u}
    combinations = []
    for v in range(num_basis_functions):
        for mu in range(num_basis_functions):
            for lambda_ in range(num_basis_functions):
                for sigma in range(num_basis_functions):
                    # Coulomb attraction two-electron integral
                    coulomb = [v,mu,sigma,lambda_]
                    # Exchange two-electron integral
                    exchange = [v,lambda_,sigma,mu]
                    # Avoid calculating identical integrals
                    # (rs|tu) = (rs|ut) = (sr|tu) = (sr|ut) = (tu|rs) = (tu|sr) = (ut|rs) = (ut|sr)
                    coulomb = tuple(sorted(coulomb[0:2]) + sorted(coulomb[2:4]))
                    exchange = tuple(sorted(exchange[0:2]) + sorted(exchange[2:4]))
                    coulomb_swapped = tuple(coulomb[2:4] + coulomb[0:2])
                    exchange_swapped = tuple(exchange[2:4] + coulomb[0:2])
                    if coulomb not in combinations and coulomb_swapped not in combinations:
                        combinations.append(coulomb)
                    if exchange not in combinations and exchange_swapped not in combinations:
                        combinations.append(exchange)

    console_print('  Two-Electron Integral Combinations (total=%d):' % len(combinations))
    for combination in combinations:
        console_print('    (%d, %d, %d, %d)' % (combination[0], combination[1], combination[2], combination[3]))

#
# Pre-calculate integrals
#
def precalculate_integrals():

    do_he_integrals = False
    do_h2_integrals = False

    try:
        with open(HE_INTEGRALS_FILENAME) as he_integrals_json_file:
            he_integrals = json.load(he_integrals_json_file)
            console_print('** HE integrals loaded from %s' % (HE_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load He integrals file, will recalculate')
        do_he_integrals = True

    try:
        with open(H2_INTEGRALS_FILENAME) as h2_integrals_json_file:
            h2_integrals = json.load(h2_integrals_json_file)
            console_print('** H2 integrals loaded from %s' % (H2_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load H2 integrals file, will recalculate')
        do_h2_integrals = True

    # dictionary keys
    OVERLAP = 'overlap'
    KINETIC = 'kinetic'
    ATTRACTION = 'attraction'
    EXCHANGE = 'exchange'

    # setup integral dictionaries
    he_integrals = {}
    h2_integrals = {}

    # *******************
    # *** HELIUM ATOM ***
    # *******************

    if do_he_integrals:

        console_print('** Starting Helium atom integral calculations')

        # Helium uses the same basis function twice, so the integrals end up being
        # identical four all four entries 11 = 12 = 21 = 22

        # generate unique combinations
        combinations = get_one_electron_combinations(HE_NUM_BASIS_FUNCTIONS)

        # basis function lookup table
        he_basis_func_lut = (sto_1g_helium_func, sto_1g_helium_func)

        console_print('Calculating Helium atom overlap integrals...')
        he_integrals[OVERLAP] = calculate_overlap_integrals(he_basis_func_lut, combinations)

        console_print('Calculating Helium atom kinetic energy integrals...')
        he_integrals[KINETIC] = calculate_kinetic_energy_integrals(he_basis_func_lut, combinations)

        console_print('Calculating Helium atom nuclear attraction integrals...')
        he_integrals[ATTRACTION] = calculate_nuclear_attraction_integrals(he_basis_func_lut, combinations, 'he')

        console_print('Calculating Helium atom repulsion and exchange integrals...')
        combinations = get_two_electron_combinations(HE_NUM_BASIS_FUNCTIONS)
        he_integrals[EXCHANGE] = calculate_coulomb_repulsion_and_exchange_integrals(he_basis_func_lut, combinations)

        # write out integrals
        console_print('** Saving Helium atom integrals to %s...' % (HE_INTEGRALS_FILENAME))
        with open(HE_INTEGRALS_FILENAME, 'w') as he_integrals_json_file:
            json.dumps(he_integrals, indent=4)

        console_print('** Finished calculating Helium atom integrals!')

    # *************************
    # *** HYDROGEN MOLECULE ***
    # *************************

    if do_h2_integrals:

        console_print('** Starting Hydrogen molecule integral calculations')

        # Hydrogen molecule has two basis functions centered on each nuclei, for a
        # total of four basis function (only two are unique). They will need to
        # be combined to form the matrices, so we'll have to find unique
        # combinations

        # basis function lookup table
        h2_basis_func_lut = (sto_1g_hydrogen_0_func, sto_1g_hydrogen_0_func, sto_1g_hydrogen_1_func, sto_1g_hydrogen_1_func)

        # generate unique combinations
        combinations = get_one_electron_combinations(H2_NUM_BASIS_FUNCTIONS)

        console_print('  Calculating Hydrogen molecule overlap integrals')
        h2_integrals[OVERLAP] = calculate_overlap_integrals(h2_basis_func_lut, combinations)

        console_print('  Calculating Hydrogen molecule kinetic energy integrals')
        h2_integrals[KINETIC] = calculate_kinetic_energy_integrals(h2_basis_func_lut, combinations)

        console_print('  Calculating Hydrogen molecule kinetic energy integrals')
        h2_integrals[ATTRACTION] = calculate_nuclear_attraction_integrals(h2_basis_func_lut, combinations, 'h2')

        console_print('  Calculating Hydrogen molecule repulsion and exchange integrals...')
        combinations = get_two_electron_combinations(H2_NUM_BASIS_FUNCTIONS)
        h2_integrals[EXCHANGE] = calculate_coulomb_repulsion_and_exchange_integrals(h2_basis_func_lut, combinations)

        # write out integrals
        console_print('** Saving Hydrogen molecule integrals to %s...' % (H2_INTEGRALS_FILENAME))
        with open(H2_INTEGRALS_FILENAME, 'w') as h2_integrals_json_file:
            json.dumps(h2_integrals, indent=4)

        console_print('** Finished calculating Hydrogen molecule integrals!')

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