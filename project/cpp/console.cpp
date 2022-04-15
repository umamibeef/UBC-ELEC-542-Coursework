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
#include <iostream>
#include <iomanip>
#include <ctime>
#include "console.hpp"

// Boost includes
#include <boost/format.hpp>

using namespace boost;

static const std::string sim_string = ANSI_FG_COLOR_CYAN         " DSHF |" ANSI_COLOR_RESET;
static const std::string cuda_string = ANSI_FG_COLOR_GREEN       "CUDART|" ANSI_COLOR_RESET;
static const std::string cusolver_string = ANSI_FG_COLOR_GREEN   "CUSOLV|" ANSI_COLOR_RESET;
static const std::string lapack_string = ANSI_FG_COLOR_YELLOW    "LAPACK|" ANSI_COLOR_RESET;

static const std::string info_string =                           "INFO|";
static const std::string warning_string = ANSI_FG_COLOR_YELLOW   "WARN|" ANSI_COLOR_RESET;
static const std::string error_string = ANSI_FG_COLOR_RED        "ERR!|" ANSI_COLOR_RESET;

static const int CONSOLE_WIDTH = 120;

void console_print_internal(int verbose_level, std::string input_string, client_e client, level_e level)
{
    if (verbose_level > program_verbosity)
    {
        return;
    }

    time_t time_now = time(nullptr);
    char time_string[100];
    strftime(time_string, sizeof(time_string), "%Y/%m/%d-%H:%M:%S", localtime(&time_now));
    std::string client_string;
    std::string level_string;

    switch (client)
    {
        case CLIENT_SIM:
            new (&client_string) std::string(sim_string);
            break;
        case CLIENT_LAPACK:
            new (&client_string) std::string(lapack_string);
            break;
        case CLIENT_CUDA:
            new (&client_string) std::string(cuda_string);
            break;
        case CLIENT_CUSOLVER:
            new (&client_string) std::string(cusolver_string);
            break;
        case CLIENT_NONE:
            break;
        default:
            new (&client_string) std::string(sim_string);
            break;
    }

    switch (level)
    {
        case LEVEL_INFO:
            new (&level_string) std::string(info_string);
            break;
        case LEVEL_WARNING:
            new (&level_string) std::string(warning_string);
            break;
        case LEVEL_ERROR:
            new (&level_string) std::string(error_string);
            break;
        case LEVEL_RULE:
            // Rule is a special case, clear the client string if we have one.
            // Input string is ignored.
            break;
        case LEVEL_NONE:
            break;
        default:
            new (&level_string) std::string(info_string);
            break;
    }

    std::string date_label = str(format("|%s|") % time_string);

    if (level != LEVEL_RULE)
    {
        std::stringstream ss(input_string);
        std::string output_string;
        while (getline(ss, output_string, '\n'))
        {
            std::cout << date_label << client_string << level_string << " " << output_string << std::endl;
        }
    }
    else
    {
        std::cout << date_label << client_string << std::string(static_cast<int>(CONSOLE_WIDTH - date_label.length() - client_string.length()),'-') << std::endl;
    }

}

void console_print(int verbose_level, std::string input_string, client_e client)
{
    console_print_internal(verbose_level, input_string, client, LEVEL_INFO);
}

void console_print_warn(int verbose_level, std::string input_string, client_e client)
{
    console_print_internal(verbose_level, input_string, client, LEVEL_WARNING);
}

void console_print_err(int verbose_level, std::string input_string, client_e client)
{
    console_print_internal(verbose_level, input_string, client, LEVEL_ERROR);
}

void console_print_spacer(int verbose_level, client_e client)
{
    console_print_internal(verbose_level, " ", client, LEVEL_NONE);
}

void console_print_hr(int verbose_level, client_e client)
{
    console_print_internal(verbose_level, " ", client, LEVEL_RULE);
}