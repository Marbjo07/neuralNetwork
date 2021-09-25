#pragma once

#include "NeuralNetwork.hpp"

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

    for (auto i = 0; i < output.size(); i++)
        e += std::abs(target[i] - output[i]);
    return e / output.size();
}

float NeuralNet::MSELossFunction(std::vector<float> output, std::vector<float> target) {
    float e{};
    for (auto i = 0; i < output.size(); i++)
        e += std::pow(target[i] - output[i], 2);
    return e;
}

// For every weight adds a random value between -1 and 1
void mergeWithRandomModel(NeuralNet* model, float mutationStrength) {
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());


    for (auto layerNum = 0; layerNum < model->m_numberLayers; layerNum++) {

        for (auto neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum++) {


            for (auto weightNum = 0; weightNum < model->m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {

                model->m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum] += ((static_cast<float>(gen()) / gen.max()) * 2 - 1) * mutationStrength;

            }

            model->m_layers[layerNum].m_neurons[neuronNum].m_bias += (static_cast<float>(gen()) / gen.max() * 2 - 1) * mutationStrength;

        }

    }
}

void NeuralNet::naturalSelection(
    std::vector<float> target,
    int numberOfTest,
    float mutationStrength,
    NeuralNet* checkerModel
) {

    NeuralNet bestModel = *this;
    NeuralNet tempModel = bestModel;
    float lowestError;
    float tempError;

    if (checkerModel != NULL) {
        checkerModel->setInput(feedForward());
        lowestError = MAELossFunction(checkerModel->feedForward(), target);

    }
    else {
        lowestError = MAELossFunction(bestModel.feedForward(), target);
    }

    for (auto gen = 0; gen < numberOfTest; gen++) {

       // auto t1 = std::chrono::high_resolution_clock::now();

        tempModel = bestModel;

        mergeWithRandomModel(&tempModel, mutationStrength);


        std::vector<float> output = tempModel.feedForward();

        if (checkerModel != NULL) {
            checkerModel->setInput(output);


            tempError = MAELossFunction(checkerModel->feedForward(), target);
        }
        else {
            tempError = MAELossFunction(output, target);
        }


        if (tempError < lowestError) {
            bestModel = tempModel;
            std::cout << "New best model: " << gen << " ";


            std::cout << "with ";
            if (checkerModel != NULL) {
                printVector(checkerModel->m_layers[checkerModel->m_numberLayers - 1].getActivation());
            }
            else {
                printVector(output);
            }
            std::cout << " as output and " << tempError << " as error." << std::endl;

            if (tempError == 0) {
                *this = bestModel;
                break;
            }
            lowestError = tempError;

            // start new generation

        }

        else {
            //std::cout << gen << " error: " << tempError << " lowest error: " << lowestError << " \n";
        }

        //std::cout << "Duration in miliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;


    }
    std::cout << sumOfWeightsAndBias() << std::endl;
    *this = bestModel;
    std::cout << sumOfWeightsAndBias() << std::endl;


}

#endif // NATURALSELECTION_CPP