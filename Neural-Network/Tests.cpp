#pragma once

#include "Tests.hpp"

namespace Test {
    namespace Private {
        std::string Test::Private::passNotPass(int returnVal) {

            if (returnVal == 0) { return "   x    |            |\n"; }
                                  return  "       |      x     |\n";
        }

        bool caEqual(float a, float b) {
            return abs(a - b) < 0.5;
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

                
                for (uint32_t N = 0; N < model.m_shape[L]; ++N) {
                    float tmp = 0;
                    for (uint32_t W = 0; W < model.m_shape[L - 1]; W++) {

                        tmp += prevActivation[W] * weights[N * model.m_shape[L - 1] + W];
                    
                    }

                    activation[N] = ACTIVATION_FUNCTION_CPU(tmp) + model.m_layers[L].m_bias;
                }

                cudaMemcpy(model.m_layers[L].d_activations, activation.data(), model.m_shape[L] * sizeof(float), cudaMemcpyHostToDevice);
            }
            return activation;
        }


        int FeedForwardTest(bool debug) {


            NeuralNet testModel;

            for (auto s = 0; s < 5; s++) {

                testModel.m_shape = { (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1, (uint32_t)std::rand() % 100 + 1 };
                
                //for (auto x : testModel.m_shape) std::cout << x << " ";

                testModel.init("FeedForwardTest", std::rand() );

                testModel.setRandomInput(std::rand());

                //std::vector<float> expectedResults;
                //expectedResults.resize(42, 96.71145246);
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

                        if (!caEqual(output, expectation)) {

                            printf("Faild at (neuron): (%d)  Iteration: %d \nOutput: %.6f  Expectation: %.6f\n", n, k, output, expectation);
                            for (auto i = 0; i < testModel.m_layers.back().m_numberNeurons; i++) std::cout << expectedResults[i] << " ";
                            printf("\n");

                            testModel.printActivations();
                            //testModel.printWeightsAndBias();
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
    }
    void run(bool exitOnFail, bool debug) {

        std::cout << "                  | " << " Passed | Didnt pass |\n";
        int output = Private::FeedForwardTest(debug);
        std::cout << "Feed Forward Test | " << Private::passNotPass(output);
        if (output != 0 && exitOnFail) { return; }
        printf("\n");
    }

  
    //FeedForwardBenchmark
    void FeedForwardBenchmark(std::vector<uint32_t> shape) {

        printf("[Feedforward Benchmark]: ");

        srand((uint64_t)time(NULL));

        NeuralNet model;


        model.m_shape = shape;

        model.init("AI", std::rand());

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

    void MutateFunctionBenchmark(std::vector<uint32_t> shape) {

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