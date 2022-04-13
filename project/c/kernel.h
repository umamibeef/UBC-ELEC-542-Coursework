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

#include "main.hpp"
#include "config.hpp"

int cuda_get_device_info(void);

int cuda_allocate_integration_memory(LutVals_t &lut_vals, DynamicDataPointers_t &ddp);
int cuda_allocate_eigensolver_memory(LutVals_t &lut_vals, DynamicDataPointers_t &ddp);
int cuda_free_integration_memory(LutVals_t &lut_vals, DynamicDataPointers_t &ddp);
int cuda_free_eigensolver_memory(DynamicDataPointers_t &ddp);

int cuda_numerical_integration(LutVals_t lut_vals, DynamicDataPointers_t ddp);
bool cuda_eigensolver(LutVals_t lut_vals, DynamicDataPointers_t ddp);