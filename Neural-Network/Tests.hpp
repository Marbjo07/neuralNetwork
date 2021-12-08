#pragma once

#include "NeuralNetwork.hpp"

#ifndef TESTS_HPP
#define TESTS_HPP

namespace Test {
    namespace Private {
        
        std::string passNotPass(int returnVal);

        bool caEqual(float a, float b);

        std::vector<float> matrixMul(NeuralNet* model);

        int FeedForwardTest(bool debug);
    }
    void run(bool exitOnFail, bool debug);


    //FeedForwardBenchmark
    void FeedForwardBenchmark();

    // One data point is one feedforward call excludes the highest data point
    //
    // model shape = {3, 256, 1024, 4096, 4096, 1024, 256, 3}
    // 
    // Feedforward()                            | Data points in microseconds                                               |   Total |  Average |
    // CPU                                      | 974888 995912 975568 993463 1023107 1004664 994783 1040392 1000714 996192 | 9999683 | 999968.3 |
    // GPU not correct                          |  13804  13760  13829  14235   13666   13674  13884   13773   13782  13736 |  138143 |  13814.3 |
    // GPU correct                              | 115582 121807 134584 120241  120700  120283 127042  124897  134610 124214 | 1243960 |  124396  |
    // GPU changed structure of neuralNet class |  39762  40986  42278  39139   37990   37074  39181   39773   36604  37715 |  390502 |  39050.2 |


    void InitBenchmark();
    
    // One data point is one call to Init()
    // 
    // model shape = {3, 256, 1024, 4096, 4096, 1024, 256, 3}
    // 
    // Init function | Data point in microseconds                                           |  Total time | Average | Change
    // v1            |221301 252704 200037 200320 201713 198987 198042 198444 199069 202674 | 2.07329e+06 |  207329 |


    void MergeFunctionBenchmark();

    // One data point is the average of 50 test with 0.05 as mutationStrength excludes the highest data point
    //
    // models shape = {3, 256, 256, 256, 256, 256, 256, 3}
    // 
    // Merge function | Data point in microseconds    | Total time | Average | Change
    // v1             | 34611 34167 34216 35725 34166 |   172885   | 34577   | Normal loop.
    // v2             | 27874 27784 28145 27764 27422 |   138989   | 27797.8 | The loop is parsely written out increments of 8.
    // v3             | 24141 23017 22812 22887 24152 |   117009   | 23401.8 | Calls a function that mutates one neuron at a time and the loop is parsely written out increments of 8.
    // v4             | 22068 21191 21492 21099 21070 |   106920   | 21384   | Pointer to random generatior instead of copying it.
    // v5             | 19081 19240 18679 18718 18640 |    94358   | 18871.6 | Saved random number generators max in a uint32_t varible.
    // v6             | 19933 19168 17493 17337 17739 |    91670   | 18334   | Changed the math. From (2r - 1)m where r = random / random max and m equals mutationStrength to (r-1)m where r = random / random max / 2 and m equals mutationStrength.
    // v7             | 19032 17991 18020 17442 18056 |    90541   | 18108.2 | Changed the math. To randomNumber * constVal - mutationStrength. Where constVal = (2 / randomNumber.max) * mutationStrength.
    // v8             | 18040 16701 17250 17065 16690 |    85746   | 17149.2 | Made a varible at the start of merge function equal to randomNumber * constVal - mutationStrength. Where constVal = (2 / random max) * mutationStrength.
    // v9             |   813   866   801   835   911 |     4226   |   845.2 | Custom random number generator






};

#endif // !TESTS_HPP */