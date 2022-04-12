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

#include <vector>
#include <iostream>

#define NO_TIME (-1.0)

/**
 * @brief      This class describes a performance monitor.
 */
class PerformanceMonitor
{
public:

    typedef enum
    {
        ITERATION_EIGENSOLVER_TIME = 0,
        ITERATION_INTEGRATION_TIME,
        ITERATION_TOTAL_TIME,
        ITERATION_TOTAL_ENERGY,
        ITERATION_NUM,
    } iteration_counter_e;

    float total_time;
    int num_iterations;
    std::vector<std::vector<float>> iteration_counters;

    /**
     * @brief      Constructs a new instance of the PerformanceMonitor class
     */
    PerformanceMonitor()
    {
        total_time = NO_TIME;
        num_iterations = 0;
        // Resize vector to support all iteration counters
        iteration_counters.resize(ITERATION_NUM);
        // First iteration needs a spot for data
        for (int i = 0; i < ITERATION_NUM; i++)
        {
            iteration_counters[i].push_back(NO_TIME);
        }
    }

    /**
     * @brief      On a new iteration, this function should be called to add a spot for new data
     */
    void next_iteration(void)
    {
        // New iteration, push back a new spot for data
        for (int i = 0; i < ITERATION_NUM; i++)
        {
            iteration_counters[i].push_back(NO_TIME);
        }
        num_iterations++;
    }

    /**
     * @brief      Record a new entry for the specified counter
     *
     * @param[in]  counter  The counter to record in
     * @param[in]  value    The value we want to record
     */
    void record(iteration_counter_e counter, float value)
    {
        iteration_counters[static_cast<unsigned int>(counter)].back() = value;
    }

    /**
     * @brief      Returns the performance monitor's records as a string
     *
     * @return     The formatted string
     */
    std::string str(void)
    {
        std::stringstream ss;

        for (int i = 0; i < num_iterations; i++)
        {
            ss  << "  Iteration " << i << "" << std::endl
                << "    Eigensolver time: " << iteration_counters[ITERATION_EIGENSOLVER_TIME][i] << " seconds" << std::endl
                << "    Integration time: " << iteration_counters[ITERATION_INTEGRATION_TIME][i] << " seconds" << std::endl
                << "    Total time: " << iteration_counters[ITERATION_TOTAL_TIME][i] << " seconds" << std::endl
                << "    Total energy: " << iteration_counters[ITERATION_TOTAL_ENERGY][i] << " Hartrees" << std::endl
                ;
        }
        ss << "Total execution time: " << total_time << " seconds" << std::endl;

        return ss.str(); 
    }

};
