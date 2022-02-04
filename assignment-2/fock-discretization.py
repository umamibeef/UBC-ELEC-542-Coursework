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
    # N = 3
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # |-6 1 0 | 1 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(0,0,0)
    # | 1-6 1 | 0 1 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(1,0,0)
    # | 0 1-6 | 0 0 1 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(2,0,0)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 1 0 0 |-6 1 0 | 1 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(0,1,0)
    # | 0 1 0 | 1-6 1 | 0 1 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(1,1,0)
    # | 0 0 1 | 0 1-6 | 0 0 1 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(2,1,0)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 1 0 0 |-6 1 0 | 1 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(0,2,0)
    # | 0 0 0 | 0 1 0 | 1-6 1 | 0 1 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 0 0 | 0 0 0 | f(1,2,0)
    # | 0 0 0 | 0 0 1 | 0 1-6 | 0 0 1 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 0 | 0 0 0 | f(2,2,0)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 1 0 0 | 0 0 0 | 1 0 0 |-6 1 0 | 1 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 0 0 0 | f(0,0,1)
    # | 0 1 0 | 0 0 0 | 0 1 0 | 1-6 1 | 0 1 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 0 0 | f(1,0,1)
    # | 0 0 1 | 0 0 0 | 0 0 1 | 0 1-6 | 0 0 1 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 0 | f(2,0,1)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 1 0 0 | 0 0 0 | 1 0 0 |-6 1 0 | 1 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | f(0,1,1) +
    # | 0 0 0 | 0 1 0 | 0 0 0 | 0 1 0 | 1-6 1 | 0 1 0 | 0 0 0 | 0 1 0 | 0 0 0 | f(1,1,1) *
    # | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 1 | 0 1-6 | 0 0 1 | 0 0 0 | 0 0 1 | 0 0 0 | f(2,1,1) &
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 1 0 0 |-6 1 0 | 0 0 0 | 0 0 0 | 1 0 0 | f(0,2,1)
    # | 0 0 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 1 0 | 1-6 1 | 0 0 0 | 0 0 0 | 0 1 0 | f(1,2,1)
    # | 0 0 0 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 1 | 0 1-6 | 0 0 0 | 0 0 0 | 0 0 1 | f(2,2,1)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 0 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 0 0 0 |-6 1 0 | 1 0 0 | 0 0 0 | f(0,0,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 0 0 | 1-6 1 | 0 1 0 | 0 0 0 | f(1,0,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 0 | 0 1-6 | 0 0 1 | 0 0 0 | f(2,0,2)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 1 0 0 |-6 1 0 | 1 0 0 | f(0,1,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 1 0 | 1-6 1 | 0 1 0 | f(1,1,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 1 | 0 1-6 | 0 0 1 | f(2,1,2)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 1 0 0 | 0 0 0 | 1 0 0 |-6 1 0 | f(0,2,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 1 0 | 0 0 0 | 0 1 0 | 1-6 1 | f(1,2,2)
    # | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 0 | 0 0 1 | 0 0 0 | 0 0 1 | 0 1-6 | f(2,2,2)
    # +-------+-------+-------+-------+-------+-------+-------+-------+-------+
    # + f(1,1,1) + f(0,2,1) + f(0,1,2) -6f(0,1,1) +          + f(0,0,1) + f(0,1,0)
    # * f(2,1,1) + f(1,2,1) + f(1,1,2) -6f(1,1,1) + f(0,1,1) + f(1,0,1) + f(1,1,0)
    # &          + f(2,2,1) + f(2,1,2) -6f(2,1,1) + f(1,1,1) + f(2,0,1) + f(2,1,0)

def main():

if __name__ == "__main__":
    main()
