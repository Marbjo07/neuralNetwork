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

    void FeedForwardBenchmark(std::vector<uint32_t> shape);

    void InitBenchmark(std::vector<uint32_t> shape);
    
    void MutateBenchmark(std::vector<uint32_t> shape);
};

#endif // !TESTS_HPP */