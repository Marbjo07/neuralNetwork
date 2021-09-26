#pragma once

#include "NeuralNetwork.hpp"

int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hi world! How are you doing?" << std::endl;

    srand(time(NULL));

    std::string savePath = "E:/desktop/neuralNet/model2.bin";

    int numberOfGenerations = 1000;
    float mutationStrength = 0.05;

    NeuralNet model;

    bool makeNewModel = true;


    if (makeNewModel) {

        model.addLayer(3);
        model.addLayer(256);
        model.addLayer(256);
        model.addLayer(256);
        model.addLayer(256);
        model.addLayer(256);
        model.addLayer(256);
        model.addLayer(3);
        
        model.init("AI");

#define NUMBERTEST 100
        

        for (auto i = 0; i < NUMBERTEST; i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            model.feedForward();
            std::cout << "Duration in miliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

        }


        //model.naturalSelection({ 1, 0, 0}, numberOfGenerations, mutationStrength);

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

    std::cout << "Duration in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

    return 0;

}
//     | Data points              | Total | Average |
// CPU | 3437 3483 3460 3420 3389 | 17189 | 3437.8  |
// GPU | Coming soon.... 


//                    | Data points              | Total | Average |
// Old merge function | 2660 2691 2545 2743 2724 | 13363 | 2672.6  |
// New merge function | 2333 2469 2291 2250 2315 | 11658 | 2331.6  |

/*void mergeWithRandomModel(NeuralNet* model, float mutationStrength) {
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());


    for (auto layerNum = 0; layerNum < model->m_numberLayers; layerNum++) {

        for (auto neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum++) {

            if (model->m_layers[layerNum].m_neurons[neuronNum].m_weights.size() > 8)
            {
                for (auto weightNum = 0; weightNum < model->m_layers[layerNum].m_neurons[neuronNum].m_weights.size() - 8; weightNum += 8) {

                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 1] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 2] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 3] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 4] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 5] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 6] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;
                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 7] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;

                }
            }
            else {
                for (auto weightNum = 0; weightNum < model->m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {

                    model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;

                }
            }
            model->m_layers[layerNum].m_neurons[neuronNum].m_bias += (static_cast<float>(gen()) / gen.max() * 2 - 1) * mutationStrength;

        }

    }
}*/