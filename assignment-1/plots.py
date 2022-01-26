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

# potential energy function for an infinitely-deep potential well 10 atomic units wide
def potential_energy_func(x):
    if abs(x) > 5:
        # return np.Inf
        return 1e10
    else:
        return 0

# colomb repulsion energy potential
def colomb_repulsion_func(x1, x2):
    if x1 == x2:
        return 1e10
    else:
        return 1.0/math.sqrt((x2-x1)**2)

# the analytically derived 1-D wave function
def wave_func(a, n, x):

    if (n % 2) == 0:
        # even
        return (2.0/math.sqrt(20.0*a))*math.sin((math.pi*n*x)/(10*a))
    else:
        # odd
        return (2.0/math.sqrt(20.0*a))*math.cos((math.pi*n*x)/(10*a))

# the analytical derived 1-D wave function calculated for a range of x coordinates
def wave_func_fromiter(offset, a, n, x_coords):
    return np.fromiter((wave_func(a, n, xi) + offset for xi in x_coords), x_coords.dtype)

def main():

    """
    Analytical plot
    """

    x_coords = np.linspace(-5, 5, 1000)
    y_coords_set = []
    for i in range(6):
        y_coords_set.append(wave_func_fromiter(i+1, 1, i+1, x_coords))

    fig, ax = plt.subplots(1, 1, gridspec_kw={'width_ratios':[1], 'height_ratios':[1]})
    for i, y_coords in enumerate(y_coords_set):
        ax.plot(x_coords, y_coords)
        ax.axhline(y=i+1, linewidth=0.5, linestyle='dashed')
    ax.axvline(x=0, linewidth=0.5, linestyle='dashed')
    ax.set_ylim([0.5,6.5])
    ax.set_xlim([-5,5])
    ax.set_title('$\Phi_n(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$n$')
    ax.spines['top'].set_visible(False)

    fig.set_size_inches(3,6)
    fig.tight_layout()

    fig.savefig('analytical-wave-function-plot.pgf')
    fig.savefig('analytical-wave-function-plot.png')

    """
    Numerical solution 1
    """

    start = -5
    end = 5
    N = 10
    print('Numerical solution 1: start=%d, end=%d, N=%d' % (start, end, N))

    # create our x-axis coordinates array using our limits
    x_coords = np.linspace(start, end, N)
    print('X coordinates:')
    print(x_coords)
    # get delta X
    delta_x = x_coords[1] - x_coords[0]

    # generate kinetic energy matrix
    kinetic_energy_matrix = np.zeros((N,N))
    # fill in the kinetic energy matrix with the coefficients of the
    # second-order centered difference approximation
    for i in range(N):
        for j in range(N):
            if i==j:
                # diagonal is -2 for the -2\Phi(x) term
                kinetic_energy_matrix[i,j]= -2
            elif np.abs(i-j)==1:
                # right outside the diagonal we have 1 on either side for the \Phi(x + \Delta x) and \Phi(x - \Delta x) terms
                kinetic_energy_matrix[i,j]=1

    # generate potential energy matrix
    potential_energy_matrix = np.zeros((N,N))
    # fill in the potential energy values along the diagonal using the x values for the problem
    for i in range(N):
        for j in range(N):
            if i==j:
                potential_energy_matrix[i,j]= potential_energy_func(x_coords[i])

    # construct the hamiltonian matrix for solving
    hamiltonian_matrix = -kinetic_energy_matrix/(2*delta_x**2) + potential_energy_matrix

    # solve our problem, obtain the eigenvectors and eigenvalues
    eigenvals, eigenvectors = np.linalg.eig(hamiltonian_matrix)
    print('eigenvals:' )
    print(eigenvals)
    print('eigenvectors:' )
    print(eigenvectors)
    # sort the eigenvalues from low to high and get their indices
    sorted_eigenval_indices = np.argsort(eigenvals)
    # get the first six indices, for the first six waveforms
    sorted_eigenval_indices = sorted_eigenval_indices[0:6]
    # get the eigenvalues, which will be out energies the first value will be
    # Eo, so we can divide all the values by this amount to obtain the
    # coefficient to compare with the analytical solution
    energies=(eigenvals[sorted_eigenval_indices]/eigenvals[sorted_eigenval_indices][0])
    print('energies:')
    print(energies)
    print('sorted eigenvalue indices')
    print(sorted_eigenval_indices)

    # plot the results
    fig, ax = plt.subplots(1, 1)
    ax.axvline(x=0, linewidth=0.5, linestyle='dashed')
    ax.set_ylim([0.5,6.5])
    ax.set_xlim([-5,5])
    for i in range(len(sorted_eigenval_indices)):
        y_coords = []
        y_coords = np.append(y_coords,eigenvectors[:,sorted_eigenval_indices[i]])
        y_coords = [y_coords + (i + 1) for y_coords in y_coords]
        ax.plot(x_coords, y_coords)
        ax.axhline(y=i+1, linewidth=0.5, linestyle='dashed')
    ax.set_title('$\Phi_n(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$n$')
    ax.spines['top'].set_visible(False)

    fig.set_size_inches(3,6)
    fig.tight_layout()

    fig.savefig('numerical-solution-1-wave-function-plot.pgf')
    fig.savefig('numerical-solution-1-wave-function-plot.png')

    """
    Numerical solution 2
    """
    
    start = -5
    end = 5
    N = 25
    print('Numerical solution 2: start=%d, end=%d, N=%d' % (start, end, N))

    # create our x1 and x2 axis coordinates array using our limits
    x_coords = np.linspace(start, end, N)
    print('X coordinates:')
    print(x_coords)
    # get delta X
    delta_x = x_coords[1] - x_coords[0]

    # generate kinetic energy sparse matrix
    # N = 4 example:
    # (*) phi(1,0) + phi(0,1) -4*phi(1,1) + phi(2,1) + phi(1, 2)
    # +---------+---------+---------+---------+
    # |-4 1 0 0 | 1 0 0 0 | 0 0 0 0 | 0 0 0 0 | phi(0,0)
    # | 1-4 1 0 | 0 1 0 0 | 0 0 0 0 | 0 0 0 0 | phi(1,0)
    # | 0 1-4 1 | 0 0 1 0 | 0 0 0 0 | 0 0 0 0 | phi(2,0)
    # | 0 0 1-4 | 0 0 0 1 | 0 0 0 0 | 0 0 0 0 | phi(3,0)
    # +---------+---------+---------+---------+
    # | 1 0 0 0 |-4 1 0 0 | 1 0 0 0 | 0 0 0 0 | phi(0,1)
    # | 0 1 0 0 | 1-4 1 0 | 0 1 0 0 | 0 0 0 0 | phi(1,1) (*)
    # | 0 0 1 0 | 0 1-4 1 | 0 0 1 0 | 0 0 0 0 | phi(2,1)
    # | 0 0 0 1 | 0 0 1-4 | 0 0 0 1 | 0 0 0 0 | phi(3,1)
    # +---------+---------+---------+---------+
    # | 0 0 0 0 | 1 0 0 0 |-4 1 0 0 | 1 0 0 0 | phi(0,2)
    # | 0 0 0 0 | 0 1 0 0 | 1-4 1 0 | 0 1 0 0 | phi(1,2)
    # | 0 0 0 0 | 0 0 1 0 | 0 1-4 1 | 0 0 1 0 | phi(2,2)
    # | 0 0 0 0 | 0 0 0 1 | 0 0 1-4 | 0 0 0 1 | phi(3,2)
    # +---------+---------+---------+---------+
    # | 0 0 0 0 | 0 0 0 0 | 1 0 0 0 |-4 1 0 0 | phi(0,3)
    # | 0 0 0 0 | 0 0 0 0 | 0 1 0 0 | 1-4 1 0 | phi(1,3)
    # | 0 0 0 0 | 0 0 0 0 | 0 0 1 0 | 0 1-4 1 | phi(2,3)
    # | 0 0 0 0 | 0 0 0 0 | 0 0 0 1 | 0 0 1-4 | phi(3,3)
    # +---------+---------+---------+---------+

    # Sparse matrix is generated using identiy matrices, concatenating columns
    # and rows, and adding it all together.
    main_diagonal = np.identity(N*N)*-4.0
    upper_side_diagonal = np.r_[np.c_[np.zeros((N*N-1,1)), np.identity(N*N-1)], np.zeros((1,N*N))]
    lower_side_diagonal = np.r_[np.zeros((1,N*N)), np.c_[np.identity(N*N-1), np.zeros((N*N-1,1))]]
    upper_outer_identity = np.r_[np.c_[np.zeros((N*N-N,N)),np.identity(N*N-N)], np.zeros((N, N*N))]
    lower_outer_identity = np.r_[np.zeros((N, N*N)), np.c_[np.identity(N*N-N), np.zeros((N*N-N,N))]]
    kinetic_energy_sparse_matrix = main_diagonal + upper_side_diagonal + lower_side_diagonal
    kinetic_energy_sparse_matrix = kinetic_energy_sparse_matrix + upper_outer_identity + lower_outer_identity

    # generate potential energy matrix
    potential_energy_matrix = np.zeros((N*N,N*N))
    # fill in the potential energy values along the diagonal using the x values for the problem
    for i in range(N*N):
        for j in range(N*N):
            if i==j:
                # each N block
                potential_energy_matrix[i,j] = 2*potential_energy_func(x_coords[i%N])

    # construct the hamiltonian matrix for solving
    hamiltonian_matrix = -kinetic_energy_sparse_matrix/(2*delta_x**2) + potential_energy_matrix

    # solve our problem, obtain the eigenvectors and eigenvalues
    eigenvals, eigenvectors = np.linalg.eig(hamiltonian_matrix)
    print('eigenvals:' )
    print(eigenvals)
    print('eigenvectors:' )
    print(eigenvectors)

    # now to organize our results
    # each N rows will represent the results for the other dimension's n
    # for example, the first N rows will correspond to \Phi(0,0) through \Phi(N,0)
    for n in range(N):
        # sort the eigenvalues from low to high and get their indices
        # get the first six indices, for the first six waveforms
        # add n*N to each index to properly index the original eigenvals since we slice it up
        # sorted_eigenval_indices.append([x+n*N for x in np.argsort(eigenvals[n:n+N+1][0:6])])
        sorted_eigenval_indices = np.argsort(eigenvals)
        # get the eigenvalues, which will be out energies the first value will be
        # Eo, so we can divide all the values by this amount to obtain the
        # coefficient to compare with the analytical solution
        energies = ((eigenvals[sorted_eigenval_indices]/eigenvals[sorted_eigenval_indices][0]))

    print('energies:')
    print(energies)
    print('sorted eigenvalue indices')
    print(sorted_eigenval_indices)

    # plot the results
    plots_row = 2
    plots_col = 3
    fig, axes = plt.subplots(plots_row, plots_col, sharex=True, sharey=True)
    for i in range(plots_row * plots_col):
        axes[i//plots_col,i%plots_col].set_title('n=%d' % (i + 1))
        axes[i//plots_col,i%plots_col].set_xlim([-5,5])
        axes[i//plots_col,i%plots_col].set_ylim([-5,5])
        # axes[i//plots_col,i%plots_col].contourf(x_coords, x_coords, eigenvectors[:,sorted_eigenval_indices[i]].reshape((N,N)))
        axes[i//plots_col,i%plots_col].imshow(eigenvectors[:,sorted_eigenval_indices[i]].reshape((N,N)), origin='lower', interpolation="none", extent=[start,end,start,end])
    fig.suptitle('$\Phi_n(x_1, x_2)$')
    fig.supxlabel("$x_1$")
    fig.supylabel("$x_2$")

    fig.set_size_inches(6,4)
    fig.tight_layout()

    fig.savefig('numerical-solution-2-wave-function-plot.pgf')
    fig.savefig('numerical-solution-2-wave-function-plot.png')

    """
    Numerical solution 3
    """

    # generate colomb repulsion matrix
    colomb_repulsion_matrix = np.zeros((N*N,N*N))
    # fill in the potential energy values along the diagonal using the x values for the problem
    for row in range(N*N):
        for col in range(N*N):
            if row==col:
                # each N block
                colomb_repulsion_matrix[row,col] = colomb_repulsion_func(row%N, col//N)
                # print('colomb_repulsion_func(%d,%d)' % (row%N, col//N))

    # construct the hamiltonian matrix for solving
    hamiltonian_matrix = -kinetic_energy_sparse_matrix/(2*delta_x**2) + colomb_repulsion_matrix

    # solve our problem, obtain the eigenvectors and eigenvalues
    eigenvals, eigenvectors = np.linalg.eig(hamiltonian_matrix)
    print('eigenvals:' )
    print(eigenvals)
    print('eigenvectors:' )
    print(eigenvectors)

    # now to organize our results
    # each N rows will represent the results for the other dimension's n
    # for example, the first N rows will correspond to \Phi(0,0) through \Phi(N,0)
    for n in range(N):
        # sort the eigenvalues from low to high and get their indices
        # get the first six indices, for the first six waveforms
        # add n*N to each index to properly index the original eigenvals since we slice it up
        # sorted_eigenval_indices.append([x+n*N for x in np.argsort(eigenvals[n:n+N+1][0:6])])
        sorted_eigenval_indices = np.argsort(eigenvals)
        # get the eigenvalues, which will be out energies the first value will be
        # Eo, so we can divide all the values by this amount to obtain the
        # coefficient to compare with the analytical solution
        energies = ((eigenvals[sorted_eigenval_indices]/eigenvals[sorted_eigenval_indices][0]))

    print('energies:')
    print(energies)
    print('sorted eigenvalue indices')
    print(sorted_eigenval_indices)

    # plot the results
    plots_row = 2
    plots_col = 3
    fig, axes = plt.subplots(plots_row, plots_col, sharex=True, sharey=True)
    for i in range(plots_row * plots_col):
        axes[i//plots_col,i%plots_col].set_title('n=%d' % (i + 1))
        axes[i//plots_col,i%plots_col].set_xlim([-5,5])
        axes[i//plots_col,i%plots_col].set_ylim([-5,5])
        # axes[i//plots_col,i%plots_col].contourf(x_coords, x_coords, eigenvectors[:,sorted_eigenval_indices[i]].reshape((N,N)))
        axes[i//plots_col,i%plots_col].imshow(eigenvectors[:,sorted_eigenval_indices[i]].reshape((N,N)), origin='lower', interpolation="none", extent=[start,end,start,end])
    fig.suptitle('$\Phi_n(x_1, x_2)$')
    fig.supxlabel("$x_1$")
    fig.supylabel("$x_2$")

    fig.set_size_inches(6,4)
    fig.tight_layout()

    fig.savefig('numerical-solution-3-wave-function-plot.pgf')
    fig.savefig('numerical-solution-3-wave-function-plot.png')

if __name__ == "__main__":
    main()
