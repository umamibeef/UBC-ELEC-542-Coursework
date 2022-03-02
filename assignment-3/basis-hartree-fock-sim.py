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
import signal

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
TINY_NUMBER = 1e-9
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
# dictionary keys
OVERLAP = 'overlap'
KINETIC = 'kinetic'
ATTRACTION = 'attraction'
EXCHANGE = 'exchange'

#
# Initializer for child processes to respect SIGINT
#
def initializer():
    #Ignore SIGINT in child workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
def sto_1g_helium_func(z, y, x):
    return 0.5881*sympy.exp(-0.7739*(sympy.sqrt(x**2+y**2+z**2))**2)

#
# Hydrogen STO-1G basis function (centered around Hydrogen nucleus 0)
#
def sto_1g_hydrogen_0_func(z, y, x):
    return 0.3696*sympy.exp(-0.4166*(sympy.sqrt((x - (-H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2+y**2+z**2))**2)

#
# Hydrogen STO-1G basis function (centered around Hydrogen nucleus 1)
#
def sto_1g_hydrogen_1_func(z, y, x):
    return 0.3696*sympy.exp(-0.4166*(sympy.sqrt((x - (H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2+y**2+z**2))**2)    

#
# Calculate the overlap integrals given the function lookup table and the desired combinations
#
def calculate_overlap_integrals(funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    overlap_ints = {}

    # symbolic version of the integrand
    overlap_intgd_sym = funcs[0](z, y, x)*funcs[1](z, y, x)
    # numerical version of the integrand
    overlap_intgd_num = sympy.lambdify([z, y, x], overlap_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    overlap_int_val = scipy.integrate.nquad(overlap_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return overlap_int_val

#
# Calculate the kinetic energy integrals given the function lookup table and the desired combinations
#
def calculate_kinetic_energy_integrals(funcs):

    # sympy vector x,y,z coordinate system
    R = sympy.vector.CoordSys3D('R')

    # symbolic version of the integrand
    kinetic_energy_intgd_sym = funcs[0](R.z, R.y, R.x)*(-1/2)*sympy.vector.Laplacian(funcs[1](R.z, R.y, R.x)).doit()
    # numerical version of the integrand
    kinetic_energy_intgd_num = sympy.lambdify([R.z, R.y, R.x], kinetic_energy_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    kinetic_energy_int_val = scipy.integrate.nquad(kinetic_energy_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return kinetic_energy_int_val

#
# Calculate the nuclear attraction integrals given the function lookup table and the desired combinations
#
def calculate_nuclear_attraction_integrals(subject, funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    # symbolic version of the integrand
    if subject == 'he':
        nuclear_attraction_intgd_sym = funcs[0](z, y, x) * (-2/sympy.sqrt(x**2 + y**2 + z**2)) * funcs[1](z, y, x)
    elif subject == 'h2':
        nuclear_attraction_intgd_sym = funcs[0](z, y, x) * (-1/sympy.sqrt((x - (-H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) - (-1/sympy.sqrt((x - (H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) * funcs[1](z, y, x)
    # numerical version of the integrand
    nuclear_attraction_intgd_num = sympy.lambdify([z, y, x], nuclear_attraction_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    nuclear_attraction_int_val = scipy.integrate.nquad(nuclear_attraction_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return nuclear_attraction_int_val

#
# Calculate the Coulomb repulsion and exchange integrals given the function lookup table and the desired combinations
#
def calculate_coulomb_repulsion_and_exchange_integrals(funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')
    u, v, w = sympy.symbols('u v w')

    # symbolic version of the integrand
    coulomb_repulsion_and_exchange_intgd_sym = funcs[0](z, y, x) * funcs[1](z, y, x) * (1/sympy.sqrt(TINY_NUMBER + (u-x)**2 + (v-y)**2 + (w-z)**2)) * funcs[2](w, v, u) * funcs[3](w, v, u)
    # numerical version of the integrand
    coulomb_repulsion_and_exchange_intgd_num = sympy.lambdify([z, y, x, w, v, u], coulomb_repulsion_and_exchange_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    coulomb_repulsion_and_exchange_int_val = scipy.integrate.nquad(coulomb_repulsion_and_exchange_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return coulomb_repulsion_and_exchange_int_val

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

    return combinations

#
# Integral calculator wrapper
#
def do_integrals(subject_name, basis_functions):

    integrals = {}

    # generate combinations for the one and two electron integral calculations
    one_electron_combinations = get_one_electron_combinations(len(basis_functions))
    one_electron_function_combinations = [(basis_functions[combination[0]], basis_functions[combination[1]]) for combination in one_electron_combinations]
    two_electron_combinations = get_two_electron_combinations(len(basis_functions))
    two_electron_function_combinations = [(basis_functions[combination[0]], basis_functions[combination[1]], basis_functions[combination[2]], basis_functions[combination[3]]) for combination in two_electron_combinations]

    with multiprocessing.Pool(processes = multiprocessing.cpu_count()-2, initializer=initializer) as pool:

        console_print('** Calculating %s overlap integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_overlap_integrals, one_electron_function_combinations), ascii=True))
        integrals[OVERLAP] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s kinetic energy integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_kinetic_energy_integrals, one_electron_function_combinations), ascii=True))
        integrals[KINETIC] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s nuclear attraction integrals...' % subject_name)
        func = functools.partial(calculate_nuclear_attraction_integrals, subject_name.lower())
        results = list(tqdm.tqdm(pool.imap(func, one_electron_function_combinations), ascii=True))
        integrals[ATTRACTION] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s repulsion and exchange integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_coulomb_repulsion_and_exchange_integrals, two_electron_function_combinations), ascii=True))
        integrals[EXCHANGE] = dict(zip(two_electron_combinations, results))

    console_print('** Finished calculating %s atom integrals!' % subject_name)

    return integrals

#
# Pre-process the integral dictionary for jsonification by converting tuples to strings
#
def preprocess_dict_for_json(integral_dict):

    for integral_type_key in list(integral_dict.keys()):
        for combination_key in list(integral_dict[integral_type_key]):
            integral_dict[integral_type_key][str(combination_key)] = integral_dict[integral_type_key].pop(combination_key)

    return integral_dict

#
# Pre-calculate integrals
#
def precalculate_integrals():

    do_he_integrals = False
    do_h2_integrals = False

    try:
        with open(HE_INTEGRALS_FILENAME) as he_integrals_json_file:
            he_integrals = json.load(he_integrals_json_file)
            # todo, convert string tuples to real tuples
            console_print('** HE integrals loaded from %s' % (HE_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load He integrals file, will recalculate')
        do_he_integrals = True

    try:
        with open(H2_INTEGRALS_FILENAME) as h2_integrals_json_file:
            h2_integrals = json.load(h2_integrals_json_file)
            # todo, convert string tuples to real tuples
            console_print('** H2 integrals loaded from %s' % (H2_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load H2 integrals file, will recalculate')
        do_h2_integrals = True

    # *******************
    # *** HELIUM ATOM ***
    # *******************

    if do_he_integrals:

        console_print('** Starting He atom integral calculations')

        # He uses the same basis function twice, so the integrals end up being
        # identical four all four entries 11 = 12 = 21 = 22

        # basis function lookup table
        he_basis_func_lut = (sto_1g_helium_func, sto_1g_helium_func)

        he_integrals = do_integrals('He', he_basis_func_lut)

        # write out integrals
        console_print('** Saving Helium atom integrals to %s...' % (HE_INTEGRALS_FILENAME))
        with open(HE_INTEGRALS_FILENAME, 'w') as he_integrals_json_file:
            he_integrals = preprocess_dict_for_json(he_integrals)
            print(he_integrals)
            json.dump(he_integrals, he_integrals_json_file, indent=4)

    # *************************
    # *** HYDROGEN MOLECULE ***
    # *************************

    if do_h2_integrals:

        console_print('** Starting H2 molecule integral calculations')

        # H2 molecule has two basis functions centered on each nuclei, for a
        # total of four basis function (only two are unique). They will need to
        # be combined to form the matrices, so we'll have to find unique
        # combinations

        # basis function lookup table
        h2_basis_func_lut = (sto_1g_hydrogen_0_func, sto_1g_hydrogen_0_func, sto_1g_hydrogen_1_func, sto_1g_hydrogen_1_func)

        h2_integrals = do_integrals('H2', h2_basis_func_lut)

        # write out integrals
        console_print('** Saving H2 molecule integrals to %s...' % (H2_INTEGRALS_FILENAME))
        with open(H2_INTEGRALS_FILENAME, 'w') as h2_integrals_json_file:
            h2_integrals = preprocess_dict_for_json(h2_integrals)
            json.dump(h2_integrals, h2_integrals_json_file,  indent=4)

        console_print('** Finished calculating H2 molecule integrals!')

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