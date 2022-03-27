# Possible optimization target: pyscf, numint_uniform_grid.c
  * integration is performed by pyscf/pyscf/pbc/dft/multigrid.py
    - specifically, the eval_mat function
    - the numeric integration function name to call is constructed and stored
      in eval_fn (e.g. eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type)
    - the numerical integration is passed as the first paramter of
      libdft.NUMINT_fill2c
    - a test function exists: test_eval_mat, found in test_numint.py and test_r_numint.py
    - BLAS and LAPACK libraries are required to build pyscf (sudo apt install libblas-dev liblapack-dev)

# Python venv setup
  * Create virtual environment
    - python3 -m venv /path/to/new/virtual/environment
  * Activate the virtual environment
    - Platform        Shell             Command to activate virtual environment
    - POSIX           bash/zsh          $ source <venv>/bin/activate
    -                 PowerShell Core   $ <venv>/bin/Activate.ps1
    - Windows         cmd.exe           C:\> <venv>\Scripts\activate.bat
    -                 PowerShell        PS C:\> <venv>\Scripts\Activate.ps1
  * Package can now be built and installed within virtual environment
    - python3 setup.py build
    - python3 setup.py install
  * Virtual environment can be deactivated by typing 'deactivate' in the shell
  * More information can be found here: https://docs.python.org/3/library/venv.html

# Notes:
  * Relevant section on numerical integration: https://pyscf.org/user/dft.html#numerical-integration-grids
  * "PySCF implements several numerical integration grids, which can be tuned in KS-DFT calculations following the examples in dft/11-grid_scheme.py"
  * PySCF makes use of libxc (https://www.tddft.org/programs/libxc/)
    - "Libxc is a library of exchange-correlation and kinetic energy functionals for density-functional theory."
  * A project that implements numerical integration grids: https://theochem.github.io/horton/2.1.1/lib/pck_horton_grid.html
  * A paper talking numerical integration: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf

# TODOs
  * finalize a benchmark to run that will utilize the numerical integration
  * understand what workloads use the numerical integration routines that will be accelerated