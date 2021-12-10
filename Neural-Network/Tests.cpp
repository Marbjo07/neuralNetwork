#pragma once

#include "Tests.hpp"
#include "Macros.hpp"

namespace Test {
    namespace Private {
        std::string Test::Private::passNotPass(int returnVal) {

            if (returnVal == 0) { return "   x    |            |\n"; }
                                  return  "       |      x     |\n";
        }

        bool caEqual(float a, float b) {
            return abs(a - b) < 0.5;
        }

        int FeedForwardTest(bool debug) {

            NeuralNet testModel;

            testModel.m_shape = { 7, 2, 42 };


            testModel.init("FeedForwardTest", 1.31415);

            float input[7] = { 1, 2, 3, 4, 5, 6, 7 };
            testModel.setInput(input, 7);

            std::vector<float> expectedResults;
            expectedResults.resize(42, 96.71145246);
            /*1.31415 1.31415
            1.31415 1.31415
                1.31415 1.31415
                1.31415 1.31415
                1.31415 1.31415
                1.31415 1.31415
                1.31415 1.31415
                1.31415 1.31415


                1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415
                1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415 1.31415






                */


            float tmp = 0;
            for (auto k = 0; k < 4; k++) {
                testModel.feedForward();
                
                float output = 0;
                float expectation = 0;

                    
                for (uint32_t n = 0; n < testModel.m_layers.back().m_numberNeurons; n++) {

                    output = testModel.getOutput()[n];
                    expectation = expectedResults[n];

                    tmp += abs(output - expectation);

                    if (!caEqual(output, expectation)) {

                        printf("Faild at (neuron): (%d)  Iteration: %d \nOutput: %.6f  Expectation: %.6f\n", n, k, output, expectation);
                        for (auto i = 0; i < testModel.m_layers.back().m_numberNeurons; i++) std::cout << expectedResults[i] << " ";
                        printf("\n");

                        testModel.printActivations();
                        testModel.printWeightsAndBias();
                        return 1;

                    }
                }
            }
            if (debug) {
                std::cout << "Error: " << tmp << std::endl;
            }
            return 0;
        }
    }
    void run(bool exitOnFail, bool debug) {

        std::cout << "                  | " << " Passed | Didnt pass |\n";
        int output = Private::FeedForwardTest(debug);
        std::cout << "Feed Forward Test | " << Private::passNotPass(output);
        if (output != 0 && exitOnFail) { return; }

    }

  
    //FeedForwardBenchmark
    void FeedForwardBenchmark() {
#include <chrono>
#include <vector>
        srand((uint32_t)time(NULL));

        NeuralNet model;


        model.m_shape = { 3, 256, 1024, 4096, 4096, 1023, 256, 3 };

        model.init("AI");

        model.setRandomInput();

        float* output;


        float total = 0;
        uint32_t numberTests = 10;
        for (uint32_t i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            model.feedForward();

            output = model.getOutput();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            std::cout << duration << " ";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

        return;
    }




    void InitBenchmark() {
#include <chrono>
#include <vector>
        srand((uint32_t)time(NULL));

        NeuralNet model;


        model.m_shape = { 3, 256, 1024, 4096, 4096, 1024, 256, 3 };

        float total = 0;
        uint32_t numberTests = 10;
        for (uint32_t i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            model.init("AI");


            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            std::cout << duration << " ";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

        return;
    }

    void MergeFunctionBenchmark() {


        srand((uint32_t)time(NULL));

        int numberOfMutations = 50;
        float mutationStrength = 0.05f;

        NeuralNet model;


        model.m_shape = { 3, 256, 256, 256, 256, 256, 256, 3 };


        model.init("AI");

        for (float i = 0; i < 6; i++) {
            model.naturalSelection({ 0 }, numberOfMutations, mutationStrength, 0);
        }

    }

};