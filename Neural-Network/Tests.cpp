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

        float* matrixMul(NeuralNet* model) {

            float tmp = 0;

            for (size_t L = 1; L < (*model).m_numberLayers; L++) {

                for (size_t k = 0; k < (*model).m_layers[L].m_numberNeurons; k++) {

                    tmp = 0;

                    for (size_t i = 0; i < (*model).m_layers[L-1].m_numberNeurons; i++) {

                        tmp += (*model).m_layers[L-1].m_activation[i] * (*model).m_layers[L].m_weights[k * (*model).m_layers[L - 1].m_numberNeurons + i];
                    }
                    (*model).m_layers[L].m_activation[k] = ACTIVATION_FUNCTION_CPU(tmp) + (*model).m_layers[L].m_bias[k];

                }
            }

            return (*model).m_layers.back().m_activation;
        }

        int FeedForwardTest(bool debug) {

            NeuralNet testModel;

            testModel.m_shape = { 7, 732, 43, 34 };


            testModel.init("FeedForwardTest");

            testModel.setRandomInput();

            std::vector<float> expectedResults(testModel.m_layers.back().m_numberNeurons);
            memcpy(&expectedResults[0], matrixMul(&testModel), testModel.m_layers.back().m_numberNeurons * sizeof(float));


            float tmp = 0;
            for (auto k = 0; k < 1; k++) {
            
                testModel.feedForward();
                
                float output = 0;
                float expectation = 0;

                    
                for (uint32_t n = 0; n < testModel.m_layers.back().m_numberNeurons; n++) {

                    output = testModel.m_layers.back().m_activation[n];
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


            output = model.feedForward();

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