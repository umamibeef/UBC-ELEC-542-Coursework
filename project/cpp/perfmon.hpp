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
#include <ctime>

#include "config.hpp"

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

    /**
     * @brief      Returns the performance monitor's records as a csv string (only the header)
     *
     * @return     The formatted string
     */
    std::string str_csv_header(void)
    {
        std::stringstream ss;

        ss  << "Time," << "Atomic," << "Threads,"
            << "GPU Int," << "GPU Eig," <<"Parts," << "Limit,"
            << "Dim," << "Step," << "Conv,"
            << "Max Its," << "It," << "Total E,"
            << "Eig Time," << "Int Time," << "It Time," << "Total Time," << std::endl;

        return ss.str();
    }

    /**
     * @brief      Returns the performance monitor's records as a csv string
     *
     * @param      cfg   A reference to the configuration structure
     * @param      lut   A reference to the lookup value structure
     *
     * @return     The formatted string
     */
    std::string str_csv_data_all(Cfg_t &config, Lut_t &lut)
    {
        std::time_t time_now = std::time(nullptr);
        char time_string[100];
        std::strftime(time_string, sizeof(time_string), "%Y/%m/%d-%H:%M:%S", std::localtime(&time_now));
        std::stringstream ss;

        for (int i = 0; i < num_iterations; i++)
        {
            ss  << time_string << "," << config.atomic_structure << "," << config.max_num_threads << ","
                << config.enable_cuda_integration << "," << config.enable_cuda_eigensolver << ","
                << config.num_partitions << "," << config.limit << ","
                << lut.matrix_dim << "," << lut.step_size << "," << config.convergence_percentage << ","
                << config.max_iterations << "," << i << ","
                << iteration_counters[PerformanceMonitor::ITERATION_TOTAL_ENERGY][i] << ","
                << iteration_counters[PerformanceMonitor::ITERATION_EIGENSOLVER_TIME][i] << ","
                << iteration_counters[PerformanceMonitor::ITERATION_INTEGRATION_TIME][i] << ","
                << iteration_counters[PerformanceMonitor::ITERATION_TOTAL_TIME][i] << ","
                << total_time << std::endl;
        }

        return ss.str();
    }

    /**
     * @brief      Returns the performance monitor's records as a string,
     *             averaged as a single entry.
     *
     * @param      cfg   A reference to the configuration structure
     * @param      lut   A reference to the lookup value structure
     *
     * @return     The formatted string averaged
     */
    std::string str_csv_data_average(Cfg_t &config, Lut_t &lut)
    {
        std::time_t time_now = std::time(nullptr);
        char time_string[100];
        std::strftime(time_string, sizeof(time_string), "%Y/%m/%d-%H:%M:%S", std::localtime(&time_now));
        std::stringstream ss;
        float total_energy_avg = 0;
        float eigensolver_time_average = 0;
        float integration_time_average = 0;
        float total_iteration_time_avg = 0;

        for (int i = 0; i < num_iterations; i++)
        {
            total_energy_avg += iteration_counters[PerformanceMonitor::ITERATION_TOTAL_ENERGY][i];
            eigensolver_time_average += iteration_counters[PerformanceMonitor::ITERATION_EIGENSOLVER_TIME][i];
            integration_time_average += iteration_counters[PerformanceMonitor::ITERATION_INTEGRATION_TIME][i];
            total_iteration_time_avg += iteration_counters[PerformanceMonitor::ITERATION_TOTAL_TIME][i];
        }

        total_energy_avg /= num_iterations;
        eigensolver_time_average /= num_iterations;
        integration_time_average /= num_iterations;
        total_iteration_time_avg /= num_iterations;

        ss  << time_string << "," << config.atomic_structure << "," << config.max_num_threads << ","
            << config.enable_cuda_integration << "," << config.enable_cuda_eigensolver << ","
            << config.num_partitions << "," << config.limit << ","
            << lut.matrix_dim << "," << lut.step_size << "," << config.convergence_percentage << ","
            << config.max_iterations << "," << "0" << ","
            << total_energy_avg << ","
            << eigensolver_time_average << ","
            << integration_time_average << ","
            << total_iteration_time_avg << ","
            << total_time << "," << std::endl;

        return ss.str();
    }

};