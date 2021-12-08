#pragma once

#include "NeuralNetwork.hpp"

#include <curand_kernel.h>
#include <curand.h>

#ifndef NATURALSELECTION_CPP
#define NATURALSELECTION_CPP

template<typename T>
void printVector(std::vector<T> vector) {
    for (auto i : vector) {
        std::cout << i << " | ";
    }
}


float NeuralNet::MAELossFunction(std::vector<float> output, std::vector<float> target) {
    float e{};

    for (auto i = 0; i < output.size() && i < target.size(); i++)
        e += std::abs(target[i] - output[i]);
    return e / output.size();
}

float NeuralNet::MSELossFunction(std::vector<float> output, std::vector<float> target) {
    float e{};
    for (auto i = 0; i < output.size() && i < target.size(); i++) {
        e += (float)std::pow(target[i] - output[i], 2);
    }
    return e;
}




// For every weight and bias adds a random value between -1 and 1
void mergeWithRandomModel(NeuralNet* model, float mutationStrength) {
    std::cout << "Sum: " << model->sumOfWeightsAndBias() << " -> ";

    for (uint32_t layerNum = 0; layerNum < model->m_numberLayers; layerNum++) {


        uint32_t weightNum = 0;
        if (model->m_layers[layerNum].m_weights.size() > 8)
        {
            for (; weightNum < model->m_layers[layerNum].m_weights.size() - 8; weightNum += 8) {

                model->m_layers[layerNum].m_weights[weightNum + 0] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 1] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 2] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 3] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 4] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 5] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 6] += Random::Default() * mutationStrength;
                model->m_layers[layerNum].m_weights[weightNum + 7] += Random::Default() * mutationStrength;
            }

        }
        else {
            for (; weightNum < model->m_layers[layerNum].m_weights.size(); weightNum++) {

                model->m_layers[layerNum].m_weights[weightNum] += Random::Default() * mutationStrength;

            }
        }

        for (uint32_t neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum++) {
            model->m_layers[layerNum].m_bias[neuronNum] += Random::Default() * mutationStrength;
        }
    }

    std::cout << model->sumOfWeightsAndBias() << "   \t\t";


}

void NeuralNet::naturalSelection(
    std::vector<float> target,
    int numberOfTest,
    float mutationStrength,
    float quitThreshold,
    NeuralNet* checkerModel
) {
    NeuralNet bestModel = *this;
    NeuralNet tempModel = bestModel;
    float lowestError{};
    float tempError{};

    auto t1 = std::chrono::high_resolution_clock::now();

    if (checkerModel != NULL) {
        checkerModel->setInput(feedForward());
        lowestError = MSELossFunction(checkerModel->feedForward(), target);

    }
    else {
        lowestError = MSELossFunction(bestModel.feedForward(), target);
    }

    if (lowestError <= quitThreshold) {
        std::cout << "Sum of weight and bias: " << sumOfWeightsAndBias() << std::endl;
        return;

    }

    std::cout << "Orgin error: " << lowestError << std::endl;

    for (auto gen = 0; gen < numberOfTest; gen++) {

        mergeWithRandomModel(&tempModel, mutationStrength);

        std::vector<float> output = tempModel.feedForward();

        if (checkerModel != NULL) {
            checkerModel->setInput(output);

            tempError = MSELossFunction(checkerModel->feedForward(), target);
        }
        else {
            tempError = MSELossFunction(output, target);
        }

        if (tempError < lowestError) {
            bestModel = tempModel;
            lowestError = tempError;
            std::cout << "New best model: " << gen << " ";


            std::cout << "with ";
            if (checkerModel != NULL) {
                printVector(checkerModel->m_layers.back().m_activation);
            }
            else {
                printVector(output);
            }

            std::cout << " as output and " << tempError << " as error." << std::endl;
            
            
            if (lowestError <= quitThreshold) {
                *this = bestModel;
                std::cout << "Sum of weight and bias: " << sumOfWeightsAndBias() << std::endl;
                return;
            
            }
        }

        else {
            std::cout << "Error: " << tempError << " lowest error: " << lowestError << std::endl;
        }



    }
    *this = bestModel;
    std::cout << "Sum of weight and bias: " << sumOfWeightsAndBias() << std::endl;


}

#endif // NATURALSELECTION_CPP