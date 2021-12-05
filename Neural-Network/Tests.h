#pragma once

#include "NeuralNetwork.hpp"

#ifndef TESTS_CPP
#define TESTS_CPP

namespace Test {
    namespace Private {
        std::string passNotPass(int returnVal) {

            if (returnVal == 0) { return "   x    |            |\n"; }
                                  return  "       |      x     |\n";
        }

        bool caEqual(float a, float b) {
            return abs(a - b) < 0.0005;
        }

        std::vector<std::vector<float>> matrixMul(NeuralNet* model) {

            float tmp = 0;

            for (uint32_t L = 0; L < (*model).m_numberLayers - 1; L++) {

                for (uint32_t k = 0; k < (*model).m_layers[L + 1].m_numberNeurons; k++) {

                    tmp = 0;

                    for (uint32_t i = 0; i < (*model).m_layers[L].m_numberNeurons; i++) {

                        tmp += (*model).m_layers[L].m_neurons[i].m_activation * (*model).m_layers[L + 1].m_neurons[k].m_weights[i];
                    }

                    (*model).m_layers[L + 1].m_neurons[k].m_activation = (*model).m_layers[L + 1].activationFunction(tmp) + (*model).m_layers[L + 1].m_neurons[k].m_bias;
                }
            }

            std::vector<std::vector<float>> out;

            out.resize((*model).m_numberLayers);
            for (auto i = 0; i < out.size(); i++) {
                out[i] = (*model).m_layers[i].getActivations();
            }



            return out;
        }

        int FeedForwardTest(bool debug) {

            NeuralNet testModel;

            testModel.m_shape = { 432, 364, 24, 64, 128};


            testModel.init("FeedForwardTest");

            testModel.setRandomInput();

            //int counter = 0;
            //for (uint32_t layerNum = 0; layerNum < model.m_numberLayers; layerNum++) {
            //
            //    for (uint32_t neuronNum = 0; neuronNum < model.m_layers[layerNum].m_numberNeurons; neuronNum++) {
            //
            //        for (uint32_t weightNum = 0; weightNum < model.m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {
            //
            //            model.m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum] = counter;
            //            counter++;
            //        }
            //
            //    }
            //
            //}

            std::vector<std::vector<float>> expectedResults = matrixMul(&testModel);
            float tmp = 0;
            for (auto k = 0; k < 3; k++) {
            
                std::vector<float> somthing = testModel.feedForward();
                
                float output = 0;
                float expectation = 0;

                for (int l = 0; l < testModel.m_numberLayers; l++) {
                    
                    for (int n = 0; n < testModel.m_layers[l].m_numberNeurons; n++) {

                        output = testModel.m_layers[l].m_neurons[n].m_activation;
                        expectation = expectedResults[l][n];
                        
                        tmp += abs(output - expectation);

                        if (!caEqual(output, expectation)) {
                            
                            printf("Faild at (layer, neuron): (%d, %d)  Iteration: %d \nOutput: %.6f  Expectation: %.6f\n", l, n, k, output, expectation);
                            //for (auto x : expectedResults) std::cout << x << " ";
                            //printf("\n");
                            //
                            //model.printActivations();
                            //printf("\n\n\n\n");
                            //model.printWeightsAndBias();
                            return 1;

                        }
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

        std::vector<float> output;


        float total = 0;
        int numberTests = 10;
        for (int i = 0; i < numberTests; i++) {
            auto start = std::chrono::high_resolution_clock::now();


            output = model.feedForward();

            float duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
            total += duration;

            // Total: 7.58581e+06 Average: 75858.1
            std::cout << duration << " ";
        }
        std::cout << "Total: " << total << " Average: " << total / numberTests << std::endl;

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