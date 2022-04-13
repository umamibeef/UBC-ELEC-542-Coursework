# Notes:
  * CUDA on WSL
    - Following instructions here: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

# TODOs
  * Immediate:
    - Use cudaMalloc and copy data vs cudaMallocManaged, because it doesn't work with large value instatiations for some reason
      - Can probably make all the necessary changes within the CUDA files and only call the CPU alloc stuff on the main program side
    - ~~Move dynamic memory allocs into their own struct so they can be more easily passed into functions that use them~~
    - ~~Add CSV outputter for performance data and results~~
    - Clean up eigensolver on CUDA side to properly take into account error conditions
    - Document with doxygen and generate
    - ~~Implement a switch between single core and multi core CPU for that additional datapoint~~ sort of implemented via shell script
  * Longterm:
    - Implement a molecule parser to generalize the program

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