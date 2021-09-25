#pragma once

#include "NeuralNetwork.hpp"

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hello World" << std::endl;

    srand(time(NULL));

    std::string savePath = "E:/desktop/neuralNet/model2.bin";

    int numberOfGenerations = 100000;
    float mutationStrength = 0.075;

    NeuralNet model;

    bool makeNewModel = true;


    if (makeNewModel) {


        model.addLayer(3);
        model.addLayer(3);
        model.addLayer(3);
        
        model.init("AI");


        model.naturalSelection({ 1,0,0 }, numberOfGenerations, mutationStrength);

        std::cout << "Natural selection done\n";

        model.save(savePath);
    }

    else {
        model.load(savePath);
    }


    std::vector<float> output = model.feedForward();

    std::cout << "Output: ";
    for (auto x : output) std::cout << x << " | ";
    std::cout << "\n";

    model.printWeightAndBias();

    std::cout << "Duration in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

    return 1;

}
//     | Data points              | Total | Average |       
// CPU | 3437 3483 3460 3420 3389 | 17189 | 3437.8  |
// GPU | Coming soon....