/** 
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
**/

#pragma once

typedef enum
{
    IDX_X = 0,
    IDX_Y = 1,
    IDX_Z = 2,
    IDX_NUM,
} coordinate_index_e;

typedef enum
{
    HELIUM_ATOM = 0,
    HYDROGEN_MOLECULE,
    ATOMIC_STRUCTURE_NUM,
} atomic_structure_e;

typedef struct
{
    int num_partitions; // Number of partitions/quantizations for the grid
    int matrix_dim; // The resulting dimensions of the solution matrices
    int limit; // The x,y,z limits of the solution space
    double step_size; // The resulting step size for the given number of partitions and limits
    int max_iterations; // Maximum number of HF interations to perform
    int num_solutions; // The number of eigenvalues and eigenvectors to keep from the solution
    float convergence_percentage; // The convergence percentage for total energy change, used for terminating the main loop
    // lookup tables to speed up calculations
    std::vector<std::vector<float>> coordinate_value_array;
    std::vector<std::vector<float>> coordinate_index_array;
} cfg_t;

#define AU_DISTANCE (5.29e-11) // Atomic unit of distance = Bohr radius (m)
#define H2_BOND_LENGTH_ATOMIC_UNITS (0.74e-10/AU_DISTANCE) // Bond length of Hydrogen atoms in Hydrogen molecule in atomic units
#define TINY_NUMBER (0.01) // use this number instead if the denominators become too small to help with convergence
#define EPSILON (TINY_NUMBER) // any number below this limit will be considered 0
#define Z_OFFSET (0.5) // add a z offset to the molecular structures to help with convergence
#define Y_OFFSET (0) // add a z offset to the molecular structures to help with convergence
#define PROGRAM_VERBOSITY (2) // Maximum logging verbosity level