#include "NeuralNetwork.h"

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hello World" << std::endl;

    srand(time(NULL));

    std::string savePath = "E:/desktop/neuralNet/model2.bin";

    int numberOfGenerations = 10000;
    int sizeOfGeneration = 100;

    NeuralNet model;

    bool makeNewModel = true;


    if (makeNewModel) {


        model.addLayer(3);
        model.addLayer(3);
        model.addLayer(3);
        
        model.init();

        model.setInputs({ 1,0,0 });

        model.naturalSelection({ 1,0,0 }, numberOfGenerations, sizeOfGeneration);

        std::cout << "Natural selection done\n";

        model.save(savePath);
    }

    else {
        model.load(savePath);
    }

    model.setInputs({ 1,0,0 });

    std::vector<float> output = model.feedForward();

    std::cout << "Output: ";
    for (auto x : output) std::cout << x << " | ";
    std::cout << "\n";

    auto t2 = std::chrono::high_resolution_clock::now();


    model.printWeightAndBias();

    std::cout << "Duration in miliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

    return 1;

}