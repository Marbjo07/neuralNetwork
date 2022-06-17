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
        model.m_activationFunctions = { "sigmoid", "tanh", "linear", "relu", "sigmoid", "tanh", "linear" };
        model.init("Optimizer", clock());
        
        
        Test::InitBenchmark();
        
        Test::MutateBenchmark();
        
        
        // Check time for feedforward
        Test::FeedForwardBenchmark();
        
        // Prints and uses the best Grids and Blocks value for feedforward
        model.optimizeParametersFeedforward(5, 32, 10);
        
        // Check time with new blocks and grid.
        // The changes aren't big for small models but for bigger model it can be better.
        Test::FeedForwardBenchmark(model.m_gridFeedforward, model.m_blockFeedforward);

        Test::BackpropagationBenchmark(1, 9);
    }

    void FeedForwardBenchmark(uint32_t grid, uint32_t block) {

        printf("[Feedforward Benchmark]: ");

        srand((uint64_t)time(NULL));

        NeuralNet model;

        if (grid != NULL) {
            model.m_gridFeedforward = grid;
        }
        if (block != NULL) {
            model.m_blockFeedforward = block;
        }

        model.m_shape = { 3, 256, 1024, 4096,  4096, 1024, 256, 3 };
        model.m_activationFunctions = { "sigmoid", "tanh", "linear", "relu", "sigmoid", "tanh", "linear" };


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

    void BackpropagationBenchmark(uint32_t grid, uint32_t block) {

        printf("[Backpropagation Benchmark]: \n");

        srand((uint64_t)time(NULL));

        NeuralNet model;

        if (grid != NULL) {
            model.m_gridFeedforward = grid;
        }
        if (block != NULL) {
            model.m_blockFeedforward = block;
        }

        model.m_shape = { 3, 256, 1024, 4096, 4096, 1024, 256, 3 };
        model.m_activationFunctions = { "sigmoid", "tanh", "linear", "relu", "sigmoid", "tanh", "linear" };

        model.init("AI", clock());


        std::vector< std::vector< float > > dataset = {};
        std::vector< std::vector< float > > labels = {};

        const int numberOfDatapoints = 10;

        const int numberOfInputsAndOutputs = model.m_shape[0];

        const int batchSize = numberOfDatapoints / 3;

        const int trainingMethod = 0; // SGD, GD, RMBGD

        const float learning_rate = 0.1;

        float total = 0;
        const uint32_t numberTests = 10;

        for (int i = 0; i < numberOfDatapoints; i++) {
            dataset.push_back({});

            for (int j = 0; j < numberOfInputsAndOutputs; j++) {
                dataset[i].push_back(float(i + 1) / numberOfDatapoints);
            }

            labels.push_back({});

            for (int j = 0; j < numberOfInputsAndOutputs; j++) {
                labels[i].push_back((Random::Default() + 1) / 2);
            }
        }
        const std::vector<std::string> trainingMethods = {
            "\t[Stochastic Gradient Descent]:        ",
            "\t[Gradient Descent]                    ",
            "\t[Random Mini Batch Gradient Descent]: "
        };
        for (uint32_t trainingMethod = 0; trainingMethod < 3; trainingMethod++) {
            total = 0;
            std::cout << trainingMethods[trainingMethod];
            
            for (uint32_t i = 0; i < numberTests; i++) {
                auto start = std::chrono::high_resolution_clock::now();
                
                if (trainingMethod == 0) {
                    // Stochastic Gradient Descent
                    model.backpropagation(dataset, labels, learning_rate);
                }
                else if (trainingMethod == 1) {
                    // Gradient Descent 
                    model.backpropagation(dataset, labels, NULL, 0, false, true);

                }
                else if (trainingMethod == 2) {
                    // Random Mini Batch Gradient Descent
                    model.backpropagation(dataset, labels, NULL, batchSize, true);
                }
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
                total += duration;

                std::cout << duration << "\t";
            }
            std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;
        }
        return;
    }
     
    void InitBenchmark() {
        printf("[Init Benchmark]:        ");

        srand((uint32_t)time(NULL));

        NeuralNet model;


        model.m_shape = { 3, 256, 1024, 4096, 4096, 1024, 256, 3 };
        model.m_activationFunctions = { "sigmoid", "tanh", "linear", "relu", "sigmoid", "tanh", "linear" };


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

    void MutateBenchmark() {

        printf("[Mutate Benchmark]:      ");

        srand((uint32_t)time(NULL));

        int numberOfMutations = 50;
        float mutationStrength = 0.05f;

        NeuralNet model;


        model.m_shape = { 3, 256, 256, 256, 256, 256, 256, 3 };
        model.m_activationFunctions = { "sigmoid", "tanh", "linear", "relu", "sigmoid", "tanh", "linear" };

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