#pragma once

#include "NeuralNetwork.cuh"

#ifndef TESTS_HPP
#define TESTS_HPP

namespace Test {
    namespace Private {
        
        std::string passNotPass(int returnVal);

        bool caEqual(float a, float b, float threshold);
    
        int MutateTest(bool debug);

        int FeedForwardTest(bool debug);

        std::vector<float> cpuVersionOfFeedforward(NeuralNet model);

    }

    void runTests(bool exitOnFail, bool debug);

    void runBenchmarks();

    void FeedForwardBenchmark(uint32_t grid = NULL, uint32_t block = NULL);

    void BackpropagationBenchmark(uint32_t grid = NULL, uint32_t block = NULL);

    void InitBenchmark();
    
    void MutateBenchmark();
};

#endif // !TESTS_HPP */