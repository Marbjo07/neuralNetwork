#include "NeuralNetwork.h"

template<typename T>
void printVector(std::vector<T> vector) {
    for (auto i : vector) {
        std::cout << i << " | ";
    }
}

float calcError(std::vector<float> output, std::vector<float> target) {
    float e{};
    for (auto i = 0; i < output.size(); i++)
        e += std::abs(target[i] - output[i]);
    return e / output.size();
}

void NeuralNet::naturalSelection(NeuralNet *pointerToOrig, std::vector<float> target, int numberOfGenerations, int sizeOfGeneration, float mutationStrenght) {
   
    NeuralNet bestModel = *pointerToOrig;
    NeuralNet tempModel;

    for (auto gen = 0; gen < numberOfGenerations; gen++) {

        tempModel = bestModel;

        float lowestError = calcError(bestModel.feedForward(), target);

        std::cout << "Gen: " << gen << std::endl;

        for (auto i = 0; i < sizeOfGeneration; i++) {

            for (auto layerNum = 0; layerNum < bestModel.m_numberLayers; layerNum++) {

                tempModel.m_layers[layerNum].mutateThisLayer(mutationStrenght);
            }
            std::vector<float> output = tempModel.feedForward();
            float tempError = calcError(output, target);

            if (tempError < lowestError) {
                std::cout << "New best model: " << gen << " " << i << " with ";
                printVector(output);
                std::cout << "as output and " << tempError << " as error." << std::endl;
                lowestError = tempError;
                bestModel = tempModel;
            }


        }
    }


    for (auto a = 0; a < m_numberLayers; a++) {
        m_layers[a].m_neurons = bestModel.m_layers[a].m_neurons;
    }

}