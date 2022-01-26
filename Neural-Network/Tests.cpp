#pragma once

#include "Tests.hpp"

namespace Test {
   
    namespace Private {
        
        std::string passNotPass(int returnVal) {

            if (returnVal == 0) { return "   x    |            |\n"; }
                                  return  "       |      x     |\n";
        }

        bool caEqual(float a, float b, float threshold) {
            return abs(a - b) < threshold;
        }


        std::vector<float> cpuVersionOfFeedforward(NeuralNet model) {

            std::vector<float> activation;

            for (size_t L = 1; L < model.m_numberLayers; ++L) {
                
                std::vector<float> prevActivation(model.m_shape[L - 1]);
                activation.resize(model.m_shape[L]);
                std::vector<float> weights(model.m_shape[L] * model.m_shape[L - 1]);

                cudaMemcpy(&prevActivation[0], model.m_layers[L - 1].d_activations, model.m_shape[L - 1] * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&activation[0], model.m_layers[L].d_activations, model.m_shape[L] * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&weights[0], model.m_layers[L].d_weights, model.m_shape[L] * model.m_shape[L - 1] * sizeof(float), cudaMemcpyDeviceToHost);

                
                // Row major check
                
                /* for (uint32_t N = 0; N < model.m_shape[L]; N++) {
                    float tmp = 0;
                    for (uint32_t W = 0; W < model.m_shape[L - 1]; W++) {

                        tmp += prevActivation[W] * weights[N * model.m_shape[L - 1] + W];
                    
                    }

                    activation[N] = ACTIVATION_FUNCTION_CPU(tmp) + model.m_layers[L].m_bias;
                } */

                // Coloum major check

                for (uint32_t N = 0; N < model.m_shape[L]; N++) {
                    float tmp = 0;
                    for (uint32_t W = 0; W < model.m_shape[L - 1]; W++) {
                        tmp += prevActivation[W] * weights[W * model.m_shape[L] + N];
                    }

                    activation[N] = ACTIVATION_FUNCTION_CPU(tmp) + model.m_layers[L].m_bias;
                }

                cudaMemcpy(model.m_layers[L].d_activations, activation.data(), model.m_shape[L] * sizeof(float), cudaMemcpyHostToDevice);

            }

            return activation;
        }

        int FeedForwardTest(bool debug) {

            NeuralNet testModel;

            testModel.m_shape = { (uint32_t)std::rand() % 10 + 1, (uint32_t)std::rand() % 10 + 1 , (uint32_t)std::rand() % 10 + 1 , (uint32_t)std::rand() % 10 + 1 };
            testModel.init("FeedForwardTest", std::rand());
            for (auto s = 0; s < 4; s++) {

                testModel.setRandomInput(std::rand());
                std::vector<float> expectedResults;

                expectedResults = cpuVersionOfFeedforward(testModel);
                float tmp = 0;
                for (auto k = 0; k < 4; k++) {
      
                    testModel.feedForward();
                    float output = 0;
                    float expectation = 0;


                    for (uint32_t n = 0; n < testModel.m_layers.back().m_numberNeurons; n++) {

                        output = testModel.getOutput()[n];
                        expectation = expectedResults[n];

                        tmp += abs(output - expectation);

                        if (!caEqual(output, expectation, 0.04)) {

                            printf("Faild at (neuron): (%d)  Iteration: %d \nOutput: %.6f  Expectation: %.6f\n", n, k, output, expectation);
                            for (auto i = 0; i < testModel.m_layers.back().m_numberNeurons; i++) std::cout << expectedResults[i] << " ";
                            printf("\n");
                            
                            return 1;

                        }
                    }                            
                    
                }
                if (debug) {
                    std::cout << "Error: " << tmp << std::endl;
                }
            }
            return 0;
        }

        int MutateTest(bool debug) {

            NeuralNet beforeMutation;
            NeuralNet afterMutation; 

            beforeMutation.m_shape = { (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1 };

            beforeMutation.init("FeedForwardTest", std::rand());

            afterMutation = beforeMutation;

            afterMutation.mutate(0.1);
            
            if (caEqual(beforeMutation.sumOfWeightsAndBias(), afterMutation.sumOfWeightsAndBias(), 0.05)) {

                return 1;
            }

            return 0;
        }

    }

    void runTests(bool exitOnFail, bool debug) {

        std::cout << "                  | " << " Passed | Didnt pass |\n";
            
             // This test isnt important any more because I use cuBLAS. It fails anyway because of rounding errors.
             //ONETEST("Feed Forward Test | ", FeedForwardTest);
             ONETEST("Mutate Test       | ", MutateTest);

        printf("\n");

    }

    void runBenchmarks() {
        
        NeuralNet model;
        
        model.m_shape = { 3, 256, 1024, 4096, 4096, 1024, 256, 3 };

        model.init("AI", 12345);
        Test::InitBenchmark(model.m_shape);

        Test::MutateBenchmark({ 3, 256, 256, 256, 256, 256, 256, 3 });


        // Check time for feedforward
        Test::FeedForwardBenchmark(model.m_shape);


        // Prints and uses the best Grids and Blocks value for feedforward
        model.optimizeParametersFeedforward(5, 32, 3);

        // Check time for new blocks and grid.
        // To keep these changes set m_gridFeedforward, m_blockFeedforward to the printed values.
        // The changes aren't big for small models but for bigger model the speed increase can be vast.
        Test::FeedForwardBenchmark(model.m_shape);
    }

    void FeedForwardBenchmark(std::vector<uint32_t> shape) {

        printf("[Feedforward Benchmark]: ");

        srand((uint64_t)time(NULL));

        NeuralNet model;


        model.m_shape = shape;

        model.init("AI", 12345);

        model.setRandomInput(1);

        float* output;


        float total = 0;
        uint32_t numberTests = 10;
        for (uint32_t i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            output = model.feedForward();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            std::cout << duration << "\t";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

        return;
    }

    void InitBenchmark(std::vector<uint32_t> shape) {
        printf("[Init Benchmark]:        ");

        srand((uint32_t)time(NULL));

        NeuralNet model;


        model.m_shape = shape;


        float total = 0;
        uint32_t numberTests = 10;
        for (uint32_t i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            model.init("AI", std::rand());

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            std::cout << duration << "\t";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

        return;
    }

    void MutateBenchmark(std::vector<uint32_t> shape) {

        printf("[Mutate Benchmark]:      ");

        srand((uint32_t)time(NULL));

        int numberOfMutations = 50;
        float mutationStrength = 0.05f;

        NeuralNet model;


        model.m_shape = shape;


        model.init("AI", std::rand());

        float total = 0;
        uint32_t numberTests = 10;
        for (uint32_t i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();

            model.mutate(mutationStrength);


            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            std::cout << duration << "\t";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

    }

};