#pragma once

#include "NeuralNetwork.cuh"

#ifndef TESTS_HPP
#define TESTS_HPP

namespace Test {
    namespace Private {
        
        std::string passNotPass(int returnVal);

        bool caEqual(float a, float b);

        int FeedForwardTest(bool debug);
    }
    void run(bool exitOnFail, bool debug);

    void FeedForwardBenchmark();

    void InitBenchmark();
    
    void MergeFunctionBenchmark();
};

#endif // !TESTS_HPP */