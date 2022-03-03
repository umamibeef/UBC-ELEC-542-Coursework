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
# This is the main function
#
def main(cmd_args):

    # prepare matrices P, S, T, V, G, F
    he_p_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))
    he_s_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))
    he_t_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))
    he_v_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))
    he_g_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))
    he_f_matrix = numpy.empty((HE_NUM_BASIS_FUNCTIONS,HE_NUM_BASIS_FUNCTIONS))

    h2_p_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))
    h2_s_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))
    h2_t_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))
    h2_v_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))
    h2_g_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))
    h2_f_matrix = numpy.empty((H2_NUM_BASIS_FUNCTIONS,H2_NUM_BASIS_FUNCTIONS))

    # read from file or pre-calculate one-electron and two-electron integrals
    he_integrals, h2_integrals = precalculate_integrals()

    # fill up S matrix
    for u in range(HE_NUM_BASIS_FUNCTIONS):
        for v in range(HE_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            he_s_matrix[u,v] = he_integrals[OVERLAP][combination]
    for u in range(H2_NUM_BASIS_FUNCTIONS):
        for v in range(H2_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            h2_s_matrix[u,v] = h2_integrals[OVERLAP][combination]

    # fill up T matrix
    for u in range(HE_NUM_BASIS_FUNCTIONS):
        for v in range(HE_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            he_t_matrix[u,v] = he_integrals[KINETIC][combination]
    for u in range(H2_NUM_BASIS_FUNCTIONS):
        for v in range(H2_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            h2_t_matrix[u,v] = h2_integrals[KINETIC][combination]

    # fill up V matrix
    for u in range(HE_NUM_BASIS_FUNCTIONS):
        for v in range(HE_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            he_v_matrix[u,v] = he_integrals[ATTRACTION][combination]
    for u in range(H2_NUM_BASIS_FUNCTIONS):
        for v in range(H2_NUM_BASIS_FUNCTIONS):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([u,v]))
            h2_v_matrix[u,v] = h2_integrals[ATTRACTION][combination]

    # obtain transformation matrix X through S^-0.5

    # calculate G matrix using density matrix P and two-electron integrals

    # calculate F = T + V + G
    # apply transform X to obtain F'
    # diagonalize F' to get C'
    # convert C' to C to get a new P
    # calculate total energy, check for convergence

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
# Helium STO-1G basis function (centered around origin)
#
def sto_1g_helium_func(z, y, x):
    return 0.5881*sympy.exp(-0.7739*(sympy.sqrt(x**2+y**2+z**2))**2)

#
# Helium STO-1G basis function class
#
class sto_1g_helium_class:

    k = 0.5881
    alpha = 0.7739

    def __init__(self, center=(0,0,0)):
        self.center = center

    def eval(self, z, y, x):
        return self.k*sympy.exp(-self.alpha*(sympy.sqrt(((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2))**2))
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
# Hydrogen STO-1G basis function class
#
class sto_1g_hydrogen_class:
    
    k = 0.3696
    alpha = 0.4166

    def __init__(self, center=(0,0,0)):
        self.center = center

    def eval(self, z, y, x):
        return self.k*sympy.exp(-self.alpha*(sympy.sqrt(((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2))**2))

#
# Calculate the overlap integrals given the function lookup table and the desired combinations
#
def calculate_overlap_integrals(funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    overlap_ints = {}

    # symbolic version of the integrand
    overlap_intgd_sym = sympy.simplify(funcs[0](z, y, x)*funcs[1](z, y, x))
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
    kinetic_energy_intgd_sym = sympy.simplify(funcs[0](R.z, R.y, R.x)*(-1/2)*sympy.vector.Laplacian(funcs[1](R.z, R.y, R.x)).doit())
    # numerical version of the integrand
    kinetic_energy_intgd_num = sympy.lambdify([R.z, R.y, R.x], kinetic_energy_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    kinetic_energy_int_val = scipy.integrate.nquad(kinetic_energy_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return kinetic_energy_int_val

#
# Calculate the nuclear attraction integrals given the function lookup table
# and the desired combinations.  This is the naive version with no
# optimizations to the integrand.
#
def calculate_nuclear_attraction_integrals_naive(subject, funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')

    # symbolic version of the integrand
    if subject == 'he':
        nuclear_attraction_intgd_sym = sympy.simplify(funcs[0](z, y, x) * (-2/sympy.sqrt(x**2 + y**2 + z**2)) * funcs[1](z, y, x))
    elif subject == 'h2':
        nuclear_attraction_intgd_sym = sympy.simplify(funcs[0](z, y, x) * (-1/sympy.sqrt((x - (-H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) + (-1/sympy.sqrt((x - (H2_BOND_LENGTH_ATOMIC_UNITS/2.0))**2 + y**2 + z**2)) * funcs[1](z, y, x))
    # numerical version of the integrand
    nuclear_attraction_intgd_num = sympy.lambdify([z, y, x], nuclear_attraction_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    nuclear_attraction_int_val = scipy.integrate.nquad(nuclear_attraction_intgd_num, [[-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf], [-scipy.inf, scipy.inf]])[0]

    return nuclear_attraction_int_val

#
# Calculate the nuclear attraction integrals given the function lookup table
# and the desired combinations. This is the optimized version to help speed
# up calculations. Note that this optmized version of the nuclear attraction
# integral optimizes into the same form as the exchange integral optimization.
#
def calculate_nuclear_attraction_integrals_optimized(subject, func_objs):

    # sympy regular symbolic variables
    u = sympy.symbols('u')

    # combine the outer K coefficients into a single one that will multiply the final result
    k_all = func_objs[0].k * func_objs[1].k

    # get the four different function alphas
    alpha = func_objs[0].alpha
    beta = func_objs[1].alpha
    gamma = 1e9 # large number to avoid infinity
    delta = 0.0

    # get the four different centers, A, B, C, D
    center_a = scipy.array(func_objs[0].center)
    center_b = scipy.array(func_objs[1].center)
    # C becomes the nuclear center
    center_c_he = scipy.array((0,0,0))
    center_c_h2_0 = scipy.array(((-H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    center_c_h2_1 = scipy.array(((H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    # unused
    center_d = scipy.array((0,0,0))

    # calculate the G coefficients
    g_ab = scipy.exp(((-alpha * beta)/(alpha + beta))*(calculate_distance(center_a, center_b)**2))
    g_cd = 1.0 # scipy.exp(((-gamma * delta)/(gamma + delta))*(calculate_distance(center_c, center_d)**2)) # this turns into 1.0 since delta = 0.0

    # calculate zeta and eta
    zeta = alpha + beta
    eta = gamma + delta

    # calculate new centers Q and P
    center_p = (((alpha)/(alpha + beta)) * center_a) + (((beta)/(alpha + beta)) * center_b)
    center_q_he = (((gamma)/(gamma + delta)) * center_c_he) + (((delta)/(gamma + delta)) * center_d)
    center_q_h2_0 = (((gamma)/(gamma + delta)) * center_c_h2_0) + (((delta)/(gamma + delta)) * center_d)
    center_q_h2_1 = (((gamma)/(gamma + delta)) * center_c_h2_1) + (((delta)/(gamma + delta)) * center_d)

    # calculate fancy v**2
    v_squared = ((zeta*eta)/(zeta + eta))

    # calculate T
    t_he = v_squared * calculate_distance(center_q_he, center_p)
    t_h2_0 = v_squared * calculate_distance(center_q_h2_0, center_p)
    t_h2_1 = v_squared * calculate_distance(center_q_h2_1, center_p)

    # combined constant
    coeff = (k_all*g_ab*g_cd)*((2*scipy.pi**(5/2))/(zeta*eta*scipy.sqrt(zeta + eta)))

    # symbolic version of the integrand
    fundamental_electron_repulsion_he_intgd_sym = coeff*sympy.exp(-t_he*u**2)
    fundamental_electron_repulsion_h2_0_intgd_sym = coeff*sympy.exp(-t_h2_0*u**2)
    fundamental_electron_repulsion_h2_1_intgd_sym = coeff*sympy.exp(-t_h2_1*u**2)
    # numerical version of the integrand
    if subject == 'he':
        fundamental_electron_repulsion_intgd_num = sympy.lambdify([u], fundamental_electron_repulsion_he_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        fundamental_electron_repulsion_int_val = scipy.integrate.nquad(fundamental_electron_repulsion_intgd_num, [[0, 1]])
    elif subject == 'h2':
        fundamental_electron_repulsion_intgd_num_0 = sympy.lambdify([u], fundamental_electron_repulsion_h2_0_intgd_sym, 'scipy')
        fundamental_electron_repulsion_intgd_num_1 = sympy.lambdify([u], fundamental_electron_repulsion_h2_1_intgd_sym, 'scipy')
        # integrate (first index of tuple contains result)
        fundamental_electron_repulsion_int_val_0 = scipy.integrate.nquad(fundamental_electron_repulsion_intgd_num_0, [[0, 1]])
        # integrate (first index of tuple contains result)
        fundamental_electron_repulsion_int_val_1 = scipy.integrate.nquad(fundamental_electron_repulsion_intgd_num_1, [[0, 1]])

    if subject == 'he':
        result = 2*fundamental_electron_repulsion_int_val[0]
    elif subject == 'h2':
        result = fundamental_electron_repulsion_int_val_0[0] + fundamental_electron_repulsion_int_val_1[0]

    return result


#
# Calculate the Coulomb repulsion and exchange integrals given the function
# lookup table and the desired combinations. This is the naive version with
# no optimizations to the integrand.
#
def calculate_coulomb_repulsion_and_exchange_integrals_naive(funcs):

    # sympy regular symbolic variables
    x, y, z = sympy.symbols('x y z')
    u, v, w = sympy.symbols('u v w')

    # nquad parameters
    limits = 1e3
    err = 1e-3

    # symbolic version of the integrand
    coulomb_repulsion_and_exchange_intgd_sym = sympy.simplify(funcs[0](z, y, x) * funcs[1](z, y, x) * (1/sympy.sqrt(TINY_NUMBER + (u-x)**2 + (v-y)**2 + (w-z)**2)) * funcs[2](w, v, u) * funcs[3](w, v, u))
    # numerical version of the integrand
    coulomb_repulsion_and_exchange_intgd_num = sympy.lambdify([z, y, x, w, v, u], coulomb_repulsion_and_exchange_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    coulomb_repulsion_and_exchange_int_val = scipy.integrate.nquad(coulomb_repulsion_and_exchange_intgd_num, [[-limits, limits]]*6, opts={'epsabs':err, 'epsrel':err}, full_output=True)
    print(coulomb_repulsion_and_exchange_int_val)

    return coulomb_repulsion_and_exchange_int_val[0]

#
# Calculate the Coulomb repulsion and exchange integrals given the function
# lookup table and the desired combinations. This is the optimized version to
# help speed up calculations.
#
def calculate_coulomb_repulsion_and_exchange_integrals_optimized(func_objs):

    # sympy regular symbolic variables
    u = sympy.symbols('u')

    # combine the outer K coefficients into a single one that will multiply the final result
    k_all = func_objs[0].k * func_objs[1].k * func_objs[2].k * func_objs[3].k

    # get the four different function alphas
    alpha = func_objs[0].alpha
    beta = func_objs[1].alpha
    gamma = func_objs[2].alpha
    delta = func_objs[3].alpha

    # get the four different centers, A, B, C, D
    center_a = scipy.array(func_objs[0].center)
    center_b = scipy.array(func_objs[1].center)
    center_c = scipy.array(func_objs[2].center)
    center_d = scipy.array(func_objs[3].center)

    # calculate the G coefficients
    g_ab = scipy.exp(((-alpha * beta)/(alpha + beta))*(calculate_distance(center_a, center_b)**2))
    g_cd = scipy.exp(((-gamma * delta)/(gamma + delta))*(calculate_distance(center_c, center_d)**2))

    # calculate zeta and eta
    zeta = alpha + beta
    eta = gamma + delta

    # calculate new centers Q and P
    center_p = (((alpha)/(alpha + beta)) * center_a) + (((beta)/(alpha + beta)) * center_b)
    center_q = (((gamma)/(gamma + delta)) * center_c) + (((delta)/(gamma + delta)) * center_d)

    # calculate fancy v**2
    v_squared = ((zeta*eta)/(zeta + eta))

    # calculate T
    t = v_squared * calculate_distance(center_q, center_p)

    # combined constant
    coeff = (k_all*g_ab*g_cd)*((2*scipy.pi**(5/2))/(zeta*eta*scipy.sqrt(zeta + eta)))

    # symbolic version of the integrand
    coulomb_repulsion_and_exchange_intgd_sym = coeff*sympy.exp(-t*u**2)
    # numerical version of the integrand
    coulomb_repulsion_and_exchange_intgd_num = sympy.lambdify([u], coulomb_repulsion_and_exchange_intgd_sym, 'scipy')
    # integrate (first index of tuple contains result)
    coulomb_repulsion_and_exchange_int_val = scipy.integrate.nquad(coulomb_repulsion_and_exchange_intgd_num, [[0, 1]])

    return coulomb_repulsion_and_exchange_int_val[0]

#
# Generate one-electron integral combinations
#
def get_one_electron_combinations(num_basis_functions):

    combinations = list(itertools.combinations_with_replacement(list(range(num_basis_functions)),2))
    console_print('  One-Electron Integral Combinations (total=%d):' % len(combinations))
    for combination in combinations:
        console_print('    (%d, %d)' % (combination[0], combination[1]))

    return combinations

#
# Generate two-electron integral combinations
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
# Distance between two 3D vectors
#
def calculate_distance(vector_0, vector_1):

    return scipy.sqrt((vector_1[0] - vector_0[0])**2 + (vector_1[1] - vector_0[1])**2 + (vector_1[2] - vector_0[2])**2)

#
# Integral calculator wrapper
#
def do_integrals(subject_name, basis_function_objs):

    integrals = {}

    # generate combinations for the one and two electron integral calculations
    # for the one-electron integrals, we use naive functions and scipy to evaluate the integrals as they are because they're simple
    one_electron_combinations = get_one_electron_combinations(len(basis_function_objs))
    one_electron_function_combinations = [(basis_function_objs[combination[0]].eval, basis_function_objs[combination[1]].eval) for combination in one_electron_combinations]
    one_electron_function_combinations_obj = [(basis_function_objs[combination[0]], basis_function_objs[combination[1]]) for combination in one_electron_combinations]

    # for the two-electron integrals, we use optimized functions which only have a single integral to evaluate as they are too complex to evaluate naively
    two_electron_combinations = get_two_electron_combinations(len(basis_function_objs))
    two_electron_function_combinations = [(basis_function_objs[combination[0]], basis_function_objs[combination[1]], basis_function_objs[combination[2]], basis_function_objs[combination[3]]) for combination in two_electron_combinations]

    with multiprocessing.Pool(processes = multiprocessing.cpu_count()-2, initializer=initializer) as pool:

        console_print('** Calculating %s overlap integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_overlap_integrals, one_electron_function_combinations), total=len(one_electron_function_combinations), ascii=True))
        integrals[OVERLAP] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s kinetic energy integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_kinetic_energy_integrals, one_electron_function_combinations), total=len(one_electron_function_combinations), ascii=True))
        integrals[KINETIC] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s nuclear attraction integrals...' % subject_name)
        func = functools.partial(calculate_nuclear_attraction_integrals_naive, subject_name.lower())
        results = list(tqdm.tqdm(pool.imap(func, one_electron_function_combinations), total=len(one_electron_function_combinations), ascii=True))
        integrals[ATTRACTION] = dict(zip(one_electron_combinations, results))

        console_print('** Calculating %s repulsion and exchange integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_coulomb_repulsion_and_exchange_integrals_optimized, two_electron_function_combinations), total=len(two_electron_function_combinations), ascii=True))
        integrals[EXCHANGE] = dict(zip(two_electron_combinations, results))

    console_print('** Finished calculating %s atom integrals!' % subject_name)

    return integrals

#
# Process the integral dictionary for jsonification by converting combination tuples to strings
#
def process_dict_for_json(integral_dict_from_program):

    for integral_type_key in list(integral_dict_from_program.keys()):
        for combination_key in list(integral_dict_from_program[integral_type_key]):
            integral_dict_from_program[integral_type_key][str(combination_key)] = integral_dict_from_program[integral_type_key].pop(combination_key)

    return integral_dict_from_program

#
# Process the integral dictionary for the program by converting combination string to tuples
#
def process_dict_for_program(integral_dict_from_json):

    for integral_type_key in list(integral_dict_from_json.keys()):
        for combination_key in list(integral_dict_from_json[integral_type_key]):
            combination_key_tuple = tuple([int(str_number) for str_number in combination_key.strip('(').strip(')').split(',')])
            integral_dict_from_json[integral_type_key][combination_key_tuple] = integral_dict_from_json[integral_type_key].pop(combination_key)

    return integral_dict_from_json

#
# Pre-calculate integrals
#
def precalculate_integrals():

    do_he_integrals = False
    do_h2_integrals = False

    try:
        with open(HE_INTEGRALS_FILENAME) as he_integrals_json_file:
            he_integrals = process_dict_for_program(json.load(he_integrals_json_file))
            console_print('** HE integrals loaded from %s' % (HE_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load He integrals file, will recalculate')
        do_he_integrals = True

    try:
        with open(H2_INTEGRALS_FILENAME) as h2_integrals_json_file:
            h2_integrals = process_dict_for_program(json.load(h2_integrals_json_file))
            console_print('** H2 integrals loaded from %s' % (H2_INTEGRALS_FILENAME))
    except:
        console_print('** Unable to load H2 integrals file, will recalculate')
        do_h2_integrals = True

    # instantiate basis function classes for integral calculations
    sto_1g_helium_obj = sto_1g_helium_class((0,0,0))
    sto_1g_hydrogen_0_obj = sto_1g_helium_class(((-H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    sto_1g_hydrogen_1_obj = sto_1g_helium_class(((H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))

    # *******************
    # *** HELIUM ATOM ***
    # *******************

    if do_he_integrals:

        console_print('** Starting He atom integral calculations')

        # He uses the same basis function twice, so the integrals end up being
        # identical four all four entries 11 = 12 = 21 = 22

        # basis function lookup table
        he_basis_func_lut = (sto_1g_helium_obj, sto_1g_helium_obj)

        he_integrals = do_integrals('He', he_basis_func_lut)

        # write out integrals
        console_print('** Saving Helium atom integrals to %s...' % (HE_INTEGRALS_FILENAME))
        with open(HE_INTEGRALS_FILENAME, 'w') as he_integrals_json_file:
            he_integrals = process_dict_for_json(he_integrals)
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
        h2_basis_func_lut = (sto_1g_hydrogen_0_obj, sto_1g_hydrogen_0_obj, sto_1g_hydrogen_1_obj, sto_1g_hydrogen_1_obj)

        h2_integrals = do_integrals('H2', h2_basis_func_lut)

        # write out integrals
        console_print('** Saving H2 molecule integrals to %s...' % (H2_INTEGRALS_FILENAME))
        with open(H2_INTEGRALS_FILENAME, 'w') as h2_integrals_json_file:
            h2_integrals = process_dict_for_json(h2_integrals)
            json.dump(h2_integrals, h2_integrals_json_file,  indent=4)

        console_print('** Finished calculating H2 molecule integrals!')

    return (he_integrals, h2_integrals)

if __name__ == '__main__':
    # the following sets up the argument parser for the program
    parser = argparse.ArgumentParser(description='Exact Hartree-Fock simulator')

    parser.add_argument('-t', type=str, default='h2', dest='target_subject', action='store', choices=['h2', 'he'],
        help='target subject to run exact HF sim on')

    parser.add_argument('-c', type=float, default=1.0, dest='convergence_percentage', action='store',
        help='percent change threshold for convergence')

    args = parser.parse_args()

    main(args)