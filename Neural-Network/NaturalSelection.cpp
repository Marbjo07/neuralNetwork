#pragma once

#include "NeuralNetwork.cuh"

#ifndef NATURALSELECTION_CPP
#define NATURALSELECTION_CPP

void printArray(float* x) {
    for (auto i = 0; i < SIZEOF(x); i++) {
        std::cout << i << " | ";
    }
}


float NeuralNet::MAELossFunction(float* output, std::vector<float> target) {
    float e{};

    for (auto i = 0; i < sizeof(output) || i < target.size(); i++)
        e += std::abs(target[i] - output[i]);
    return e / SIZEOF(output);
}

float NeuralNet::MSELossFunction(float* output, std::vector<float> target) {
    float e{};
    for (auto i = 0; i < sizeof(output) || i < target.size(); i++) {
        e += (float)std::pow(target[i] - output[i], 2);
    }
    return e;
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
        checkerModel->setInput(feedForward(), m_layers[0].m_numberNeurons);
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

        //mergeWithRandomModel(&tempModel, mutationStrength);

        float* output = tempModel.feedForward();

        if (checkerModel != NULL) {
            checkerModel->setInput(output, tempModel.m_layers.back().m_numberNeurons);

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
                printArray(checkerModel->m_layers.back().d_activations);
            }
            else {
                printArray(output);
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