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
static unsigned long x = 123456789;
static unsigned long y = 362436069;
static unsigned long z = 521288629;
float randomNumberGenerator()
{
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    // does this to get a random float between -1 and 1
    return (float)z * 4.656612873e-10F - 1;
}

void mutateNeuron(float* bias, std::vector<float>* weights, float mutationStrength, float offset) {
    uint32_t weightNum = 0;
    if (weights->size() > 8)
    {
        for (; weightNum < weights->size() - 8; weightNum += 8) {

            (*weights)[weightNum + 0] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 1] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 2] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 3] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 4] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 5] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 6] += randomNumberGenerator() * mutationStrength;
            (*weights)[weightNum + 7] += randomNumberGenerator() * mutationStrength;
        }

    }
    else {
        for (; weightNum < weights->size(); weightNum++) {

            (*weights)[weightNum] += randomNumberGenerator() * mutationStrength;

        }
    }
    (*bias) += randomNumberGenerator() * mutationStrength;

}


// For every weight and bias adds a random value between -1 and 1
void mergeWithRandomModel(NeuralNet* model, float mutationStrength) {
    std::cout << "Sum: " << model->sumOfWeightsAndBias() << " -> ";

    for (uint32_t layerNum = 0; layerNum < model->m_numberLayers; layerNum++) {
        for (uint32_t neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum++) {
            mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum].m_weights, mutationStrength, (float)rand());
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
                printVector(checkerModel->m_layers.back().getActivations());
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