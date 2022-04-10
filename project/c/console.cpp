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

// C/C++ includes
#include <sstream>
#include <ctime>
#include "console.hpp"

// Boost includes
#include <boost/format.hpp>

using namespace std;
using namespace boost;

static const string none_string = " ";
static const string sim_string = ANSI_COLOR_CYAN "[SIM]" ANSI_COLOR_RESET " ";
static const string cuda_string = ANSI_COLOR_GREEN "[CUDA]" ANSI_COLOR_RESET " ";
static const string lapack_string = ANSI_COLOR_YELLOW "[LAPACK]" ANSI_COLOR_RESET " ";
static const string error_string = ANSI_COLOR_RED "[ERROR]" ANSI_COLOR_RESET " ";

void console_print_internal(int verbose_level, std::string input_string, client_e client, bool error)
{
    if (verbose_level > program_verbosity)
    {
        return;
    }
    
    time_t time_now = time(nullptr);
    char time_string[100];
    strftime(time_string, sizeof(time_string), "%Y/%m/%d-%H:%M:%S", localtime(&time_now));
    string client_string;

    switch (client)
    {
        case SIM:
            client_string = sim_string;
            break;
        case LAPACK:
            client_string = lapack_string;
            break;
        case CUDA:
            client_string = cuda_string;
            break;
        case NONE:
        default:
            client_string = none_string;
            break;
    }

    if (error)
    {
        client_string += error_string;
    }

    stringstream ss(input_string);
    string output_string;
    while (getline(ss, output_string, '\n'))
    {
        cout << format("[%s] ") % time_string << client_string << output_string << endl;
    }
}

void console_print(int verbose_level, std::string input_string, client_e client)
{
    console_print_internal(verbose_level, input_string, client, false);
}

void console_print_err(int verbose_level, std::string input_string, client_e client)
{
    console_print_internal(verbose_level, input_string, client, true);
}