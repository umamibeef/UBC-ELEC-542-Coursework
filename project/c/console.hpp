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

extern int program_verbosity;

// Console ANSI colors
#define ANSI_FG_COLOR_RED       "\x1b[31m"
#define ANSI_FG_COLOR_GREEN     "\x1b[32m"
#define ANSI_FG_COLOR_YELLOW    "\x1b[33m"
#define ANSI_FG_COLOR_BLUE      "\x1b[34m"
#define ANSI_FG_COLOR_MAGENTA   "\x1b[35m"
#define ANSI_FG_COLOR_CYAN      "\x1b[36m"

#define ANSI_BG_COLOR_BLACK     "\x1b[40m"
#define ANSI_BG_COLOR_RED       "\x1b[41m"
#define ANSI_BG_COLOR_GREEN     "\x1b[42m"
#define ANSI_BG_COLOR_YELLOW    "\x1b[43m"
#define ANSI_BG_COLOR_BLUE      "\x1b[44m"
#define ANSI_BG_COLOR_MAGENTA   "\x1b[45m"
#define ANSI_BG_COLOR_CYAN      "\x1b[46m"
#define ANSI_BG_COLOR_WHITE     "\x1b[47m"

#define ANSI_COLOR_RESET        "\x1b[0m"
#define ERASE_SCREEN            "\x1b[2J"

#define TAB1                    "  "
#define TAB2                    "    "

typedef enum
{
    CLIENT_NONE = 0,
    CLIENT_SIM,
    CLIENT_CUDA,
    CLIENT_CUSOLVER,
    CLIENT_LAPACK,
} client_e;

typedef enum
{
    LEVEL_NONE = 0,
    LEVEL_INFO,
    LEVEL_WARNING,
    LEVEL_ERROR,
} level_e;

// Headers
#include <iostream>

// Functions
void console_print(int verbose_level, std::string input_string, client_e client);
void console_print_err(int verbose_level, std::string input_string, client_e client);
void console_print_warn(int verbose_level, std::string input_string, client_e client);
void console_print_spacer(int verbose_level, client_e client);