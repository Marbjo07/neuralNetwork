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

void mutateNeuron(float* bias, std::vector<float>* weights, float mutationStrength, std::mt19937* gen, const float constVal) {
    auto weightNum = 0;
    if (weights->size() > 8)
    {
        for (; weightNum < weights->size() - 8; weightNum += 8) {

            (*weights)[weightNum + 0] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 1] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 2] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 3] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 4] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 5] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 6] += (float((*gen)()) * constVal) - mutationStrength;
            (*weights)[weightNum + 7] += (float((*gen)()) * constVal) - mutationStrength;
        }
    }

    else {
        for (; weightNum < weights->size(); weightNum++) {

            (*weights)[weightNum] += (float((*gen)()) * constVal) - mutationStrength;

        }
    }
    (*bias) += (float((*gen)()) * constVal) - mutationStrength;
    //std::cout << (float((*gen)()) * constVal) - mutationStrength << std::endl;
}


#define STEPSIZE 8
// For every weight and bias adds a random value between -1 and 1
void mergeWithRandomModel(NeuralNet* model, float mutationStrength) {
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

    const float constVal = (2.0f / float(gen.max())) * mutationStrength;

    for (auto layerNum = 0; layerNum < model->m_numberLayers; layerNum++) {


        if (STEPSIZE < model->m_layers[layerNum].m_numberNeurons) {
            for (int neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum += STEPSIZE) {

                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 0].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 0].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 1].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 1].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 2].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 2].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 3].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 3].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 4].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 4].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 5].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 5].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 6].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 6].m_weights, mutationStrength, &gen, constVal);
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum + 7].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum + 7].m_weights, mutationStrength, &gen, constVal);

            }
        }
        else {
            for (auto neuronNum = 0; neuronNum < model->m_layers[layerNum].m_numberNeurons; neuronNum ++) {
                mutateNeuron(&model->m_layers[layerNum].m_neurons[neuronNum].m_bias, &model->m_layers[layerNum].m_neurons[neuronNum].m_weights, mutationStrength, &gen, constVal);
            }
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

    auto t1 = std::chrono::high_resolution_clock::now();

    if (checkerModel != NULL) {
        checkerModel->setInput(feedForward());
        lowestError = MSELossFunction(checkerModel->feedForward(), target);

    }
    else {
        lowestError = MSELossFunction(bestModel.feedForward(), target);
    }

    for (auto gen = 0; gen < numberOfTest; gen++) {

        tempModel = bestModel;
        auto t1 = std::chrono::high_resolution_clock::now();

        mergeWithRandomModel(&tempModel, mutationStrength);
        std::cout << "Duration in miliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;


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
            if (0) {
                std::cout << "New best model: " << gen << " ";


                std::cout << "with ";
                if (checkerModel != NULL) {
                    printVector(checkerModel->m_layers[checkerModel->m_numberLayers - 1].getActivation());
                }
                else {
                    printVector(output);
                }
                std::cout << " as output and " << tempError << " as error." << std::endl;
            }if (tempError == 0) {
                *this = bestModel;
                break;
            }
            lowestError = tempError;
        }

        else {
            std::cout << "Error: " << tempError << " lowest error: " << lowestError << std::endl;
        }



    }
    *this = bestModel;
    std::cout << "Sum of weight and bias: " << sumOfWeightsAndBias() << std::endl;


}

#endif // NATURALSELECTION_CPP