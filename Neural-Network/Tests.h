#pragma once

#include "NeuralNetwork.hpp"

#ifndef TESTS_CPP
#define TESTS_CPP

namespace Test {
    namespace Private {
        std::string passNotPass(int returnVal) {

            if (returnVal == 0) { return "   x    |            |\n"; }
            return  "       |      x     | \n";
        }

        bool caEqual(float a, float b) {
            return ((a + 0.1 >= b) && (a - 0.1 <= b));
        }

        std::vector<float> matrixMul(NeuralNet* model) {

            std::vector<float> outActivation;
            float tmp = 0;

            for (uint32_t k = 0; k < (*model).m_layers[1].m_numberNeurons; k++) {

                tmp = 0;
                
                for (uint32_t i = 0; i < (*model).m_layers[0].m_numberNeurons; i++) {

                    tmp += (*model).m_layers[0].m_neurons[i].m_activation * (*model).m_layers[1].m_neurons[k].m_weights[i];
                }

                outActivation.push_back((*model).m_layers[1].m_neurons[k].activationFunction(tmp));
            }



            return outActivation;
        }

        int FeedForwardTest(bool debug) {

            NeuralNet model;

            model.addLayer(91);
            model.addLayer(63);
            model.addLayer(1);

            model.init("FeedForwardTest");

            model.setRandomInput();

            std::vector<std::vector<float>> weights;
            model.m_layers[1].getWeights(&weights);

            std::vector<float> expectedResults = matrixMul(&model);

            for (auto k = 0; k < 4; k++) {
            
                
                model.feedForward();

                for (size_t i = 0; i < model.m_layers[1].m_numberNeurons; i++) {
                    if (!caEqual(model.m_layers[1].m_neurons[i].m_activation, expectedResults[i])) {

                        std::cout << "Faild at: " << i << " iteration: " << k << std::endl;
                        
                        for (auto x : expectedResults) std::cout << x << " ";
                        printf("\n");

                        model.printActivations();

                        return 1;
                    }
                }
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

        model.setInput({ 1, 1, 1 });

        std::vector<float> output;



        for (float i = 0; i < 10; i++) {
            auto start = std::chrono::high_resolution_clock::now();


            output = model.feedForward();

            std::cout << " + " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
        }

        return;
    }

    // One data point is one feedforward call excludes the highest data point
    //
    // model shape = {3, 256, 1024, 4096, 4096, 1023, 256, 3}
    // 
    //                 | Data points in microseconds                                               |   Total |  Average |
    // CPU             | 974888 995912 975568 993463 1023107 1004664 994783 1040392 1000714 996192 | 9999683 | 999968.3 |
    // GPU not correct |  13804  13760  13829  14235   13666   13674  13884   13773   13782  13736 |  138143 |  13814.3 |
    // GPU correct     | 115582 121807 134584 120241  120700  120283 127042  124897  134610 124214 | 1243960 | 124396   |



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


    // One data point is the average of 50 test with 0.05 as mutationStrength excludes the highest data point
    //
    // models shape = {3, 256, 256, 256, 256, 256, 256, 3}
    // 
    // Merge function | Data point in microseconds    | Total time | Average | Description 
    // v1             | 34611 34167 34216 35725 34166 |   172885   | 34577   | Normal loop.
    // v2             | 27874 27784 28145 27764 27422 |   138989   | 27797.8 | The loop is parsely written out increments of 8.
    // v3             | 24141 23017 22812 22887 24152 |   117009   | 23401.8 | Calls a function that mutates one neuron at a time and the loop is parsely written out increments of 8.
    // v4             | 22068 21191 21492 21099 21070 |   106920   | 21384   | Pointer to random generatior instead of copying it.
    // v5             | 19081 19240 18679 18718 18640 |    94358   | 18871.6 | Saved random number generators max in a uint32_t varible.
    // v6             | 19933 19168 17493 17337 17739 |    91670   | 18334   | Changed the math. From (2r - 1)m where r = random / random max and m equals mutationStrength to (r-1)m where r = random / random max / 2 and m equals mutationStrength.
    // v7             | 19032 17991 18020 17442 18056 |    90541   | 18108.2 | Changed the math. To randomNumber * constVal - mutationStrength. Where constVal = (2 / randomNumber.max) * mutationStrength.
    // v8             | 18040 16701 17250 17065 16690 |    85746   | 17149.2 | Made a varible at the start of merge function equal to randomNumber * constVal - mutationStrength. Where constVal = (2 / random max) * mutationStrength.
    // v9             |   813   866   801   835   911 |     4226   |   845.2 | Custom random number generator



    // One data point is the average of 50 test with {3, 256, 1024, 4096, 4096, 1024, 256, 3} as neuralNetwork shape
    // 
    // Init function | Data point in microseconds    | Total time | Average | Description
    // v1            | 14695 13077 13540 13074 13207 |   67593    | 13518.6 | Using std::random
    // v2            |  4899  4707  5050  4987  4614 |   24257    |  4851.4 | Same optimizion as in Merge function
    // v3            |  4293  4543  4239  4265  4313 |   21653    |  4330.6 | Weights is declared as constVal and then looped through and multiplied by random number.


};

#endif // !TESTS_CPP */