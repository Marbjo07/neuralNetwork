#include "NeuralNetwork.h"

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hello World" << std::endl;

    srand(1);


    NeuralNet model;

    bool makeNewModel = false;

    if (makeNewModel) {

        model.addLayer(3);
        model.addLayer(4);
        for (auto i = 0; i < 10000; i++) {
            model.addLayer(100);
        }

        model.addLayer(4);
        model.addLayer(3);

        model.init();

        model.setInputs({ 1,0,0 });

        model.naturalSelection(&model, { 1,0,0 }, 1, 1, 1);

        std::cout << "Natural selection done\n";

        model.save("E:/desktop/neuralNet/model2.bin");
    }

    else {
        model.load("E:/desktop/neuralNet/model2.bin");

       
    }

    model.setInputs({ 1,0,0 });

    std::vector<float> output = model.feedForward();
    std::cout << "Output: ";
    for (auto x : output) {
        std::cout << x << " | ";
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;


    return 1;

}


/*
 for (auto layerNum = 0; layerNum < model.m_numberLayers; layerNum++) {
            for (auto neuronNum = 0; neuronNum < model.m_layers[layerNum].m_numberNeurons; neuronNum++) {
                for (auto weightNum = 0; weightNum < model.m_layers[layerNum].m_neurons[neuronNum].m_weight.size(); weightNum++) {
                    std::cout << model.m_layers[layerNum].m_neurons[neuronNum].m_weight[weightNum] << " | ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }*/