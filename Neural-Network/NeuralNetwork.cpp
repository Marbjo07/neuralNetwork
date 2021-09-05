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


void NeuralNet::Layer::Neuron::mutateWeightAndBias(float mutationStrength) {
    for (auto i = 0; i < m_weight.size(); i++) {

        m_weight[i] += static_cast <float> (rand()) / RAND_MAX * 2 - 1 * mutationStrength;
    
    }
    m_bias += static_cast <float> (rand()) / RAND_MAX * 2 - 1 * mutationStrength;
}


float NeuralNet::Layer::Neuron::activationFunction(float x) {
    // sigmoid
    return (1 / (1 + pow(2.71828182846, -x))) + m_bias;
}

//
// Layer class:
//


NeuralNet::Layer::Layer(int numberOfNeurons, int numberOfNeuronsNextLayer, bool zeroWeights) {
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

std::vector<float> NeuralNet::Layer::getBias() {
    std::vector<float> a; 

    for (auto i = 0; i < m_numberNeurons; i++) {
        a.push_back(m_neurons[i].m_bias);

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
        m_neurons[i].mutateWeightAndBias(mutationStrenght);
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
        m_layers[i + 1].setActivation(
                dot(m_layers[i].getActivation(), m_layers[i].getWeights())
       );
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

    // clear layers if init is already called
    m_layers.clear();
    for (int i = 0; i < this->m_shape.size() - 1; i++) {
        m_layers.push_back(Layer(this->m_shape[i], this->m_shape[i + 1], zeroWeights));
    }
    // Adds placeholder neurons
    m_layers.push_back(Layer(this->m_shape[this->m_shape.size() - 1]));

    this->m_numberLayers = m_shape.size();
}

std::vector<float> NeuralNet::dot(std::vector<float> x1, std::vector<std::vector<float>> x2) {

    std::vector<float> output;

    float t = 0;

    for (auto i = 0; i < x2[0].size(); i++) {
        for (auto a = 0; a < x1.size(); a++) {
            t += x1[a] * x2[a][i];
        }

        output.push_back(this->m_layers[0].m_neurons[0].activationFunction(t));
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

        // Save bias
        for (auto layer : m_layers) {
            for (auto neuron : layer.m_neurons) {
                saveFile.write((const char*)&neuron.m_bias, sizeof(float));
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


        // Load value of weights
        for (auto layer = 0; layer < m_numberLayers; layer++) {
            for (auto neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
                for (auto weight = 0; weight < m_layers[layer].m_neurons[neuron].m_weight.size(); weight++) {
                    loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_weight[weight], sizeof(float));
                }
            }
        }

        // Load value of bias
        for (auto layer = 0; layer < m_numberLayers; layer++) {
            for (auto neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
        
                loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_bias, sizeof(float));
            
            }
        }

        loadFile.close();
    }

    std::cout << "\nDone loading pre trained model\n";
}



void NeuralNet::printWeightAndBias() {

    std::cout << "\n\n\n";


    std::cout << "Weight: \n";

    for (auto layerNum = 0; layerNum < m_numberLayers - 1; layerNum++) {
        for (auto neuron : m_layers[layerNum].m_neurons) {
            for (auto weight : neuron.m_weight) {
                std::cout << weight << " | ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Bias: \n";
    for (auto layer : m_layers) {
        for (auto neuron : layer.m_neurons) {
            std::cout << neuron.m_bias << " | ";
        }
        std::cout << "\n";
    }


    std::cout << "\n\n\n";

}