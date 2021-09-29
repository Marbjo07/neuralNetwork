#pragma once

#include "NeuralNetwork.hpp"

void Test() {

    auto t1 = std::chrono::high_resolution_clock::now();

    srand(time(NULL));

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    int numberOfMutations = 50000;
    float mutationStrength = 0.0001;

    NeuralNet model;

    std::cout << "Loaded model\n";


    // Adds a virtuell layer with N number of neurons.
    // 
    // models shape:
    // o 
    //   \
    // o - o - o
    //   /
    //  o 

    // o is a neuron
    // \, / or - is a connection


    model.addLayer(3);
    model.addLayer(1);
    model.addLayer(1);


    // Makes all weights and bias
    // Init will clear model if allredy called!!
    // If defualtWeight if specified every weight is set to that value
    // "AI" is the name of the model. The name is printed in warrnings
    model.init("AI");

    // Output of model
    std::vector<float> output = model.feedForward();

    // Print output
    std::cout << "Output: ";
    for (auto x : output) std::cout << x << " | ";
    std::cout << "\n";


    for (auto j = 0; j < 5; j++) {
        for (float i = 0; i < 10; i++) {

            // Sets input
            model.setInput({ i });


            // 1. Mutates the original model.
            // 2. If the mutation is better than the original the mutation is now the original.
            // 3. Error is calculated by MSE or if checkerModel is specified do step 4
            // 4. Error is calculated by MSE(target and output of checkerModel with input of output of this model)
            // 5. Do step 1 through 4 numberOfMutations times.
            model.naturalSelection({ i }, numberOfMutations, mutationStrength);


            std::cout << "Target: " << i << " Output: " << model.feedForward()[0] << std::endl;
        }
    }

    // Saves the model to a bin file.
    model.save(savePath);

    // Loads a pretraind model
    // Just make a empty model and load from bin file
    // DO NOT call init after loading model because this will clear the loaded model.
    model.load(savePath);


    output = model.feedForward();

    std::cout << "Output: ";
    for (auto x : output) std::cout << x << " | ";
    std::cout << "\n";

    // Prints sum of weights and bias used in debuging.
    std::cout << model.sumOfWeightsAndBias() << std::endl;


    std::cout << "Duration in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;
}

// One data point is one feedforward call excludes the highest data point
//
// model shape = {3, 256, 1024, 4096, 4096, 1023, 256, 3}
// 
//     | Data points in microseconds                                               |   Total |  Average |
// CPU | 974888 995912 975568 993463 1023107 1004664 994783 1040392 1000714 996192 | 9999683 | 999968.3 |
// GPU |  13804  13760  13829  14235   13666   13674  13884   13773   13782  13736 |  138143 |  13814.3 |



// One data point is the average of 50 test with 0.05 as mutationStrength excludes the highest data point
//
// models shape = {3, 256, 256, 256, 256, 256, 256, 3}
// 
// Merge function | Data point in microseconds    | Total time | Average | Description 
// v1             | 34611 34167 34216 35725 34166 | 172885     | 34577   | Normal loop.
// v2             | 27874 27784 28145 27764 27422 | 138989     | 27797.8 | The loop is parsely written out increments of 8.
// v3             | 24141 23017 22812 22887 24152 | 117009     | 23401.8 | Calls a function that mutates one neuron at a time and the loop is parsely written out increments of 8.
// v4             | 22068 21191 21492 21099 21070 | 106920     | 21384   | Pointer to random generatior instead of copying it.
// v5             | 19081 19240 18679 18718 18640 | 94358      | 18871.6 | Saved random number generators max in a uint32_t varible.
// v6             | 19933 19168 17493 17337 17739 | 91670      | 18334   | Changed the math. From (2r - 1)m where r = random / random max and m equals mutationStrength to (r-1)m where r = random / random max / 2 and m equals mutationStrength.
// v7             | 19032 17991 18020 17442 18056 | 90541      | 18108.2 | Changed the math. To randomNumber * constVal - mutationStrength. Where constVal = (2 / randomNumber.max * mutationStrength.
// v8             | 18040 16701 17250 17065 16690 | 85746      | 17149.2 | Made a varible at the start of merge function equal to randomNumber * constVal - mutationStrength. Where constVal = (2 / randomNumber.max * mutationStrength.



// One data point is the average of 50 test with {3, 256, 1024, 4096, 4096, 1024, 256, 3} as neuralNetwork shape
// 
// Init function | Data point in microseconds    | Total time | Average | Description
// v1            | 14695 13077 13540 13074 13207 |   67593    | 13518.6 | Using std::random
// v2            |  4899  4707  5050  4987  4614 |   24257    |  4851.4 | Same optimizion as in Merge function
// v3            |  4293  4543  4239  4265  4313 |   21653    |  4330.6 | Weights is declared as constVal and then looped through and multiplied by random number.