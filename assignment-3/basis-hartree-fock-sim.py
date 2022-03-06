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
import copy

numpy.set_printoptions(precision=None, suppress=None, edgeitems=30, linewidth=100000, 
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
NUM_ELECTRONS = 2
H2_BOND_LENGTH_ATOMIC_UNITS = 1.39839733222307
TINY_NUMBER = 1e-9
DATETIME_STR_FORMAT = '[%Y/%m/%d-%H:%M:%S]'
HE_NUM_BASIS_FUNCTIONS = 2
H2_NUM_BASIS_FUNCTIONS = 4
HE_INTEGRALS_FILENAME = 'he_integrals.json'
H2_INTEGRALS_FILENAME = 'h2_integrals.json'
PROGRAM_VERBOSITY = 1
# dictionary keys
OVERLAP = 'overlap'
KINETIC = 'kinetic'
ATTRACTION = 'attraction'
EXCHANGE = 'exchange'

#
# This is the main function
#
def main(cmd_args):

    # read from file or pre-calculate one-electron and two-electron integrals
    he_integrals, h2_integrals = precalculate_integrals()

    # do HF for He
    do_hartree(HE_NUM_BASIS_FUNCTIONS, he_integrals)
    # do HF for H2
    do_hartree(H2_NUM_BASIS_FUNCTIONS, h2_integrals)


def do_hartree(num_basis_functions, integrals):

    # prepare matrices P, S, T, V, G, F
    p_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    new_p_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    s_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    t_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    v_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    g_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    f_matrix = numpy.empty((num_basis_functions,num_basis_functions))
    f_prime_matrix = numpy.empty((num_basis_functions,num_basis_functions))

    # fill up S matrix
    for v in range(num_basis_functions):
        for u in range(num_basis_functions):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([v,u]))
            s_matrix[v,u] = integrals[OVERLAP][combination]

    console_print(1, 'S matrix:')
    console_print(1, str(s_matrix))

    # fill up T matrix
    for v in range(num_basis_functions):
        for u in range(num_basis_functions):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([v,u]))
            t_matrix[v,u] = integrals[KINETIC][combination]

    console_print(1, 'T matrix:')
    console_print(1, str(t_matrix))

    # fill up V matrix
    for v in range(num_basis_functions):
        for u in range(num_basis_functions):
            # sort the combination to obtain the unique integral result
            combination = tuple(sorted([v,u]))
            v_matrix[v,u] = integrals[ATTRACTION][combination]

    console_print(1, 'V matrix:')
    console_print(1, str(v_matrix))

    # obtain transformation matrix X through S^-0.5
    # obtain eigenvalues
    s_eigenvalues, s_eigenvectors = numpy.linalg.eigh(s_matrix)
    # sort eigenthings
    sorted_indices = numpy.argsort(s_eigenvalues)
    s_eigenvalues = s_eigenvalues[sorted_indices]
    s_eigenvectors = s_eigenvectors[:,sorted_indices]
    console_print(1, 'S eigenvalues:')
    console_print(1, str(s_eigenvalues))
    console_print(1, 'S eigenvectors:')
    console_print(1, str(s_eigenvectors))
    # inverse square root the resulting eigenvalues and put them in a diagonal matrix
    # add a TINY_NUMBER to avoid div by zero
    s_eigenvalues = numpy.array([eigenvalue + TINY_NUMBER for eigenvalue in s_eigenvalues])
    s_eigenvalues_inverse_square_root = numpy.diag(s_eigenvalues**-0.5)
    # form the transformation matrix X by undiagonalizing the previous matrix
    x_matrix = s_eigenvectors @ s_eigenvalues_inverse_square_root @ numpy.transpose(s_eigenvectors)

    # MAIN HF LOOP
    iteration = 0
    while True:

        console_print(0, '\n **** ITERATION %d **** \n' % iteration)

        # calculate G matrix using density matrix P and two-electron integrals
        for v in range(num_basis_functions):
            for u in range(num_basis_functions):
                for lambda_ in range(num_basis_functions):
                    for sigma in range(num_basis_functions):
                        # get index
                        # Coulomb attraction two-electron integral
                        coulomb_combination = [v,u,sigma,lambda_]
                        # Exchange two-electron integral
                        exchange_combination = [v,lambda_,sigma,u]
                        # Avoid calculating identical integrals
                        # (rs|tu) = (rs|ut) = (sr|tu) = (sr|ut) = (tu|rs) = (tu|sr) = (ut|rs) = (ut|sr)
                        coulomb_combination = tuple(sorted(coulomb_combination[0:2]) + sorted(coulomb_combination[2:4]))
                        exchange_combination = tuple(sorted(exchange_combination[0:2]) + sorted(exchange_combination[2:4]))
                        coulomb_combination_swapped = tuple(coulomb_combination[2:4] + coulomb_combination[0:2])
                        exchange_combination_swapped = tuple(exchange_combination[2:4] + exchange_combination[0:2])

                        # integral is going to exist in dictionary as _combination or _combination_swapped
                        # TODO: there's got to be a better way!
                        try:
                            coulomb_term = integrals[EXCHANGE][coulomb_combination]
                        except:
                            coulomb_term = integrals[EXCHANGE][coulomb_combination_swapped]

                        try:
                            exchange_term = integrals[EXCHANGE][exchange_combination]
                        except:
                            exchange_term = integrals[EXCHANGE][exchange_combination_swapped]

                        g_matrix[v,u] = p_matrix[lambda_,sigma]*(coulomb_term - 0.5*exchange_term)

        console_print(1, 'G matrix:')
        console_print(1, str(g_matrix))

        # calculate F = T + V + G
        f_matrix = t_matrix + v_matrix + g_matrix

        console_print(1, 'F matrix:')
        console_print(1, str(f_matrix))

        # apply transform X to obtain F'
        f_prime_matrix = numpy.transpose(x_matrix) @ f_matrix @ x_matrix

        console_print(1, 'F\' matrix:')
        console_print(1, str(f_prime_matrix))

        # diagonalize F' to get C'
        f_prime_eigenvalues, c_prime_matrix = numpy.linalg.eigh(f_prime_matrix)
        # sort eigenthings
        sorted_indices = numpy.argsort(f_prime_eigenvalues)
        f_prime_eigenvalues = s_eigenvalues[sorted_indices]
        c_prime_matrix = c_prime_matrix[:,sorted_indices]
        console_print(1, 'F\' eigenvalues:')
        console_print(1, str(f_prime_eigenvalues))
        console_print(1, 'C\' matrix:')
        console_print(1, str(c_prime_matrix))

        # convert C' to C to get a new P
        c_matrix = x_matrix @ c_prime_matrix

        console_print(1, 'C matrix:')
        console_print(1, str(c_matrix))

        console_print(1, 'old P matrix:')
        console_print(1, str(p_matrix))

        # calculate the new P matrix
        for v in range(num_basis_functions):
            for u in range(num_basis_functions):
                new_p_matrix[v,u] = 0
                for n in range(int(NUM_ELECTRONS/2)):
                    new_p_matrix[v,u] = new_p_matrix[v,u] + 2*c_matrix[v,n]*c_matrix[u,n]
                    console_print(2, 'new_p_matrix[%d,%d] = %f' % (v,u,new_p_matrix[v,u]))
                    console_print(2, 'c_matrix[%d,%d] = %f' % (v,n,c_matrix[v,n]))
                    console_print(2, 'c_matrix[%d,%d] = %f' % (u,n,c_matrix[u,n]))
                    console_print(2, 'new_p_matrix[%d,%d] = new_p_matrix[%d,%d] + 2*c_matrix[%d,%d]*c_matrix[%d,%d] = %f' % (v,u,v,u,v,n,u,n,new_p_matrix[v,u]))

        console_print(1, 'new P matrix:')
        console_print(1, str(new_p_matrix))

        # compare old and new P matrix
        # get the average percent difference of all elements
        delta = 0
        for v in range(num_basis_functions):
            for u in range(num_basis_functions):
                console_print(2, 'v=%d u=%d error: %f' % (v, u, abs(new_p_matrix[v,u] - p_matrix[v,u])/abs((new_p_matrix[v,u] + p_matrix[v,u])/2)))
                delta = delta + (p_matrix[v,u] - new_p_matrix[v,u])**2
        delta = (delta / num_basis_functions)**0.5

        # set old p matrix to new
        p_matrix = new_p_matrix

        # calculate total energy, check for convergence
        total_energy_sum = 0
        for v in range(num_basis_functions):
            for u in range(num_basis_functions):
                total_energy_sum = total_energy_sum + p_matrix[v,u]*(t_matrix[v,u] + v_matrix[v,u] + f_matrix[v,u])
        total_energy = 0.5*total_energy_sum

        console_print(0, '\t\tTotal energy: %f' % total_energy)
        console_print(0, '\t\tDelta between P matrices: %f' % delta)

        # increment iteration
        iteration = iteration + 1

        # check for end condition
        if delta < 0.00001 or iteration > 10:
            break
#
# Initializer for child processes to respect SIGINT
#
def initializer():
    #Ignore SIGINT in child workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)

#
# Console formatter
#
def console_print(verbose_level=0, string='', end='\n'):

    if verbose_level > PROGRAM_VERBOSITY:
        return

    # get str representation
    if not isinstance(string, str):
        string = str(string)

    # split string at new lines
    string = string.split('\n')

    # write out line by line
    for string_line in string:
        datetime_now = datetime.datetime.now()
        print(datetime_now.strftime(DATETIME_STR_FORMAT) + ' ' + string_line, end=end)

#
# Helium STO-1G basis function class
#
class sto_1g_helium_class:

    k = 0.5881
    alpha = 0.7739

    def __init__(self, center=(0,0,0), modifier=1.0):
        self.center = center
        self.alpha = self.alpha*modifier

    def eval(self, z, y, x):
        return self.k*sympy.exp(-self.alpha*(sympy.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2)**2))

#
# Hydrogen STO-1G basis function class
#
class sto_1g_hydrogen_class:
    
    k = 0.3696
    alpha = 0.4166

    def __init__(self, center=(0,0,0), modifier=1.0):
        self.center = center
        self.alpha = self.alpha*modifier

    def eval(self, z, y, x):
        return self.k*sympy.exp(-self.alpha*(sympy.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2)**2))

#
# Calculate the overlap integrals given the function lookup table and the desired combinations
#
def calculate_overlap_integrals_naive(funcs):

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
# Calculate the overlap integrals given the function lookup table and the desired combinations
#
def calculate_overlap_integrals_optimized(func_objs):

    # combine the outer K coefficients into a single one that will multiply the final result
    k_all = func_objs[0].k * func_objs[1].k

    # get the four different function alphas
    alpha = func_objs[0].alpha
    gamma = func_objs[1].alpha

    # get the two different centers, A, C
    center_a = scipy.array(func_objs[0].center)
    center_c = scipy.array(func_objs[1].center)

    overlap_int_val = k_all*((scipy.pi/(alpha + gamma))**(3.0/2.0))*scipy.exp(-alpha*gamma*(calculate_distance(center_a, center_c)**2)/(alpha + gamma))

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
    g_ab = numpy.exp(((-alpha * beta)/(alpha + beta))*(calculate_distance(center_a, center_b)**2))
    g_cd = 1.0 # numpy.exp(((-gamma * delta)/(gamma + delta))*(calculate_distance(center_c, center_d)**2)) # this turns into 1.0 since delta = 0.0

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
    g_ab = numpy.exp(((-alpha * beta)/(alpha + beta))*(calculate_distance(center_a, center_b)**2))
    g_cd = numpy.exp(((-gamma * delta)/(gamma + delta))*(calculate_distance(center_c, center_d)**2))

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
    console_print(0, '  One-Electron Integral Combinations (total=%d):' % len(combinations))
    for combination in combinations:
        console_print(0, '    (%d, %d)' % (combination[0], combination[1]))

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

    console_print(0, '  Two-Electron Integral Combinations (total=%d):' % len(combinations))
    for combination in combinations:
        console_print(0, '    (%d, %d, %d, %d)' % (combination[0], combination[1], combination[2], combination[3]))

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

    # generate combinations for the one and two-electron integral calculations
    # for the one-electron integrals, we use naive functions and scipy to evaluate the integrals as they are because they're simple
    one_electron_combinations = get_one_electron_combinations(len(basis_function_objs))
    one_electron_function_combinations = [(basis_function_objs[combination[0]].eval, basis_function_objs[combination[1]].eval) for combination in one_electron_combinations]
    one_electron_function_combinations_obj = [(basis_function_objs[combination[0]], basis_function_objs[combination[1]]) for combination in one_electron_combinations]

    # for the two-electron integrals, we use optimized functions which only have a single integral to evaluate as they are too complex to evaluate naively
    two_electron_combinations = get_two_electron_combinations(len(basis_function_objs))
    two_electron_function_combinations = [(basis_function_objs[combination[0]], basis_function_objs[combination[1]], basis_function_objs[combination[2]], basis_function_objs[combination[3]]) for combination in two_electron_combinations]

    calculate_overlap_integrals_optimized(one_electron_function_combinations_obj[0])

    with multiprocessing.Pool(processes = multiprocessing.cpu_count()-2, initializer=initializer) as pool:

        console_print(0, '** Calculating %s overlap integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_overlap_integrals_optimized, one_electron_function_combinations_obj), total=len(one_electron_function_combinations_obj), ascii=True))
        integrals[OVERLAP] = dict(zip(one_electron_combinations, results))

        console_print(0, '** Calculating %s kinetic energy integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_kinetic_energy_integrals, one_electron_function_combinations), total=len(one_electron_function_combinations), ascii=True))
        integrals[KINETIC] = dict(zip(one_electron_combinations, results))

        console_print(0, '** Calculating %s nuclear attraction integrals...' % subject_name)
        func = functools.partial(calculate_nuclear_attraction_integrals_naive, subject_name.lower())
        results = list(tqdm.tqdm(pool.imap(func, one_electron_function_combinations), total=len(one_electron_function_combinations), ascii=True))
        integrals[ATTRACTION] = dict(zip(one_electron_combinations, results))

        console_print(0, '** Calculating %s repulsion and exchange integrals...' % subject_name)
        results = list(tqdm.tqdm(pool.imap(calculate_coulomb_repulsion_and_exchange_integrals_optimized, two_electron_function_combinations), total=len(two_electron_function_combinations), ascii=True))
        integrals[EXCHANGE] = dict(zip(two_electron_combinations, results))

    console_print(0, '** Finished calculating %s atom integrals!' % subject_name)

    return integrals

#
# This functions returns whether or not the two matrices are within a specified tolerance
#
def is_symmetric(A, B, tol=0.1):
    return scipy.sparse.linalg.norm(A-B, scipy.inf) < tol;

#
# Process the integral dictionary for jsonification by converting combination tuples to strings
#
def process_dict_for_json(integral_dict_from_program):

    integral_dict_json_copy = copy.deepcopy(integral_dict_from_program)
    for integral_type_key in list(integral_dict_json_copy.keys()):
        for combination_key in list(integral_dict_json_copy[integral_type_key]):
            integral_dict_json_copy[integral_type_key][str(combination_key)] = integral_dict_json_copy[integral_type_key].pop(combination_key)

    return integral_dict_json_copy

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
            console_print(0, '** HE integrals loaded from %s' % (HE_INTEGRALS_FILENAME))
    except:
        console_print(0, '** Unable to load He integrals file, will recalculate')
        do_he_integrals = True

    try:
        with open(H2_INTEGRALS_FILENAME) as h2_integrals_json_file:
            h2_integrals = process_dict_for_program(json.load(h2_integrals_json_file))
            console_print(0, '** H2 integrals loaded from %s' % (H2_INTEGRALS_FILENAME))
    except:
        console_print(0, '** Unable to load H2 integrals file, will recalculate')
        do_h2_integrals = True

    # instantiate basis function classes for integral calculations
    sto_1g_helium_0_obj = sto_1g_helium_class((0,0,0))
    sto_1g_helium_1_obj = sto_1g_helium_class((0,0,0))
    sto_1g_hydrogen_00_obj = sto_1g_hydrogen_class(((-H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    sto_1g_hydrogen_01_obj = sto_1g_hydrogen_class(((-H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    sto_1g_hydrogen_10_obj = sto_1g_hydrogen_class(((H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))
    sto_1g_hydrogen_11_obj = sto_1g_hydrogen_class(((H2_BOND_LENGTH_ATOMIC_UNITS/2.0),0,0))

    # *******************
    # *** HELIUM ATOM ***
    # *******************

    if do_he_integrals:

        console_print(0, '** Starting He atom integral calculations')

        # He uses the same basis function twice, so the integrals end up being
        # identical four all four entries 11 = 12 = 21 = 22

        # basis function lookup table
        he_basis_func_lut = (sto_1g_helium_0_obj, sto_1g_helium_1_obj)

        he_integrals = do_integrals('He', he_basis_func_lut)

        # write out integrals
        console_print(0, '** Saving Helium atom integrals to %s...' % (HE_INTEGRALS_FILENAME))
        with open(HE_INTEGRALS_FILENAME, 'w') as he_integrals_json_file:
            json.dump(process_dict_for_json(he_integrals), he_integrals_json_file, indent=4)

    # *************************
    # *** HYDROGEN MOLECULE ***
    # *************************

    if do_h2_integrals:

        console_print(0, '** Starting H2 molecule integral calculations')

        # H2 molecule has two basis functions centered on each nuclei, for a
        # total of four basis function (only two are unique). They will need to
        # be combined to form the matrices, so we'll have to find unique
        # combinations

        # basis function lookup table
        h2_basis_func_lut = (sto_1g_hydrogen_00_obj, sto_1g_hydrogen_01_obj, sto_1g_hydrogen_10_obj, sto_1g_hydrogen_11_obj)

        h2_integrals = do_integrals('H2', h2_basis_func_lut)

        # write out integrals
        console_print(0, '** Saving H2 molecule integrals to %s...' % (H2_INTEGRALS_FILENAME))
        with open(H2_INTEGRALS_FILENAME, 'w') as h2_integrals_json_file:
            json.dump(process_dict_for_json(h2_integrals), h2_integrals_json_file,  indent=4)

        console_print(0, '** Finished calculating H2 molecule integrals!')

    return (he_integrals, h2_integrals)

if __name__ == '__main__':
    # the following sets up the argument parser for the program
    parser = argparse.ArgumentParser(description='Exact Hartree-Fock simulator')

    args = parser.parse_args()

    main(args)