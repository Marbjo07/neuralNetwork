#pragma once

#include "NeuralNetwork.h"

//
// Neuron class:
//

NeuralNet::Layer::Neuron::Neuron(int numberOfNeuronsNextLayer, bool zeroWeights) {

    if (zeroWeights) {
        m_weight.resize(numberOfNeuronsNextLayer);
    }
    else {
        for (auto i = 0; i < numberOfNeuronsNextLayer; i++) {
            m_weight.push_back(static_cast <float> (rand()) / RAND_MAX * 2 - 1);
        }
    }
}


void NeuralNet::Layer::Neuron::mutateWeight(float mutationStrength) {
    for (auto i = 0; i < m_weight.size(); i++) {


        // random number between 1 - mutatationStrength and 1 + mutatationStrength
        m_weight[i] *= (1.0 - mutationStrength) + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / ((1.0 + mutationStrength) - (1 - mutationStrength))));
    }
}

float NeuralNet::Layer::Neuron::sigmoid(float x) {
    return 1 / (1 + pow(2.71828182846, -x));
}

//
// Layer class:
//


NeuralNet::Layer::Layer(bool createWeights, int numberOfNeurons, int numberOfNeuronsNextLayer, bool zeroWeights) {
    m_numberNeurons = numberOfNeurons;

    for (auto i = 0; i < m_numberNeurons; i++) {
        m_neurons.push_back(Neuron(numberOfNeuronsNextLayer, zeroWeights));
    }
}

std::vector<std::vector<float>> NeuralNet::Layer::getWeights() {
    std::vector<std::vector<float>> w;

    for (auto i = 0; i < this->m_numberNeurons; i++) {
        w.push_back(m_neurons[i].m_weight);
    }

    return w;
}

std::vector<float> NeuralNet::Layer::getActivation() {
    std::vector<float> a;

    for (auto i = 0; i < m_numberNeurons; i++) {
        a.push_back(m_neurons[i].m_activation);

    }
    return a;
}

void NeuralNet::Layer::setActivation(std::vector<float> a) {
    for (auto i = 0; i < m_numberNeurons; i++) {
        m_neurons[i].m_activation = a[i];
    }
}

void NeuralNet::Layer::mutateThisLayer(float mutationStrenght) {
    for (auto i = 0; i < m_numberNeurons; i++) {
        m_neurons[i].mutateWeight(mutationStrenght);
    }
}
//
// NeuralNet class:
//


void NeuralNet::addLayer(int numberOfNeurons) {
    m_shape.push_back(numberOfNeurons);
}

std::vector<float> NeuralNet::feedForward() {

    for (auto i = 0; i < m_numberLayers - 1; i++) {
        m_layers[i + 1].setActivation(dot(m_layers[i].getActivation(), m_layers[i].getWeights()));
    }

    std::vector<float> a = m_layers[m_numberLayers - 1].getActivation();

    return a;
}

void NeuralNet::setInputs(std::vector<float> inputs) {
    if (inputs.size() != this->m_shape[0]) {
        std::cout << "\033[1;31mWARNING: \033[0m" << "In setInputs() input size not matching networks first layer.Unexpected behavior may occur." << std::endl;
        inputs.resize(this->m_shape[0]);
    }
    this->m_layers[0].setActivation(inputs);
}

void NeuralNet::init(bool zeroWeights) {
    for (int i = 0; i < this->m_shape.size() - 1; i++) {
        if (i % 100 == 0) {
            std::cout << i << std::endl;
        }
        m_layers.push_back(Layer(true, this->m_shape[i], this->m_shape[i + 1], zeroWeights));
    }
    // Adds placeholder neurons.
    m_layers.push_back(Layer(false, this->m_shape[this->m_shape.size() - 1]));

    this->m_numberLayers = m_shape.size();
}

std::vector<float> NeuralNet::dot(std::vector<float> x1, std::vector<std::vector<float>> x2) {

    std::vector<float> output;

    float t = 0;

    for (auto i = 0; i < x2[0].size(); i++) {
        for (auto a = 0; a < x1.size(); a++) {
            t += x1[a] * x2[a][i];
        }
        output.push_back(this->m_layers[0].m_neurons[0].sigmoid(t));
        t = 0;
    }


    //https://matrix.reshish.com/multCalculation.php

    return output;
}

/**
 *
  This function cant accept \
 Please use /
 *
 */
void NeuralNet::save(std::string path) {

    std::cout << "Saving... \n";

    std::ofstream saveFile(path, std::ios_base::binary);

    if (saveFile.is_open()) {

        // Save size of m_shape
        int sizeOfShape = m_shape.size();
        saveFile.write(reinterpret_cast<const char*>(&sizeOfShape), sizeof(int));


        // Save shape of model
        saveFile.write(reinterpret_cast<const char*>(&m_shape[0]), m_shape.size() * sizeof(int));

        // Save weights
        for (auto layer : m_layers) {
            for (auto neuron : layer.m_neurons) {
                for (auto weight : neuron.m_weight) {
                    saveFile.write((const char*)&weight, sizeof(float));
                }
            }
        } 
        saveFile.close();
    }

    std::cout << "Done saving!\n";

}

void NeuralNet::load(std::string path) {

    std::cout << "Loading pre trained model...\n";

    std::ifstream loadFile(path, std::ios_base::binary);

    if (loadFile.is_open()) {

        int sizeOfShape;

        // Get number of layers
        loadFile.read((char*)&sizeOfShape, sizeof(sizeOfShape));

        m_shape.resize(sizeOfShape);
        
        // Get shape of network
        loadFile.read(reinterpret_cast<char*>(&m_shape[0]), sizeOfShape * sizeof(int));
        
        // Initialize without random weights
        init(true);


        // Set value for weights
        for (auto layer = 0; layer < m_numberLayers; layer++) {
            for (auto neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
                for (auto weight = 0; weight < m_layers[layer].m_neurons[neuron].m_weight.size(); weight++) {
                    loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_weight[weight], sizeof(float));
                }
            }
        }


        loadFile.close();
    }

    std::cout << "\nDone loading pre trained model\n";
}
