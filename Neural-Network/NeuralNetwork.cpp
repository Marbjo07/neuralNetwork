#pragma once

#include "NeuralNetwork.h"



#define debuging 0

#ifdef debuging 1
#define LOG(x) x
#else
#define LOG(x)
#endif


#ifndef USER_H
#define USER_H


//
// Neuron class:
//

// Connections from this neuron to the next layers neurons
NeuralNet::Layer::Neuron::Neuron(int numberOfNeuronsNextLayer) {

    for (auto i = 0; i < numberOfNeuronsNextLayer; i++)
        m_weight.push_back(static_cast <float> (rand()) / RAND_MAX * 2 - 1);
}



//
// Layer class:
//


NeuralNet::Layer::Layer(bool createWeightsFlagg, int numberOfNeurons, int numberOfNeuronsNextLayer) {
    m_numberNeurons = numberOfNeurons;
    m_numberNeuronsNextLayer = numberOfNeuronsNextLayer;

    for (auto i = 0; i < m_numberNeurons; i++) {
        m_neurons.push_back(Neuron(m_numberNeuronsNextLayer));
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


//
// NeuralNet class:
//



void NeuralNet::addLayer(int numberOfNeurons) {

    m_shape.push_back(numberOfNeurons);

    LOG(std::cout << "Shape of network: ";
    for (auto i : m_shape)
        std::cout << i << " | ";
    std::cout << std::endl;)

        this->m_numberLayers += 1;
}

std::vector<float> NeuralNet::feedForward() {

    for (auto i = 0; i < m_numberLayers - 1; i++) {
        m_layers[i + 1].setActivation(dot(m_layers[i].getActivation(), m_layers[i].getWeights()));
    }


    LOG(std::cout << "FeedForward() working" << std::endl);
    return m_layers[m_numberLayers - 1].getActivation();
}


void NeuralNet::setInputs(std::vector<float> inputs) {
    if (inputs.size() != this->m_shape[0]) {
        std::cout << "\033[1;31mWARNING: \033[0m" << "In setInputs() input size not matching networks first layer.Unexpected behavior may occur." << std::endl;
        inputs.resize(this->m_shape[0]);
    }
    this->m_layers[0].setActivation(inputs);
}

void NeuralNet::init() {
    for (int i = 0; i < this->m_shape.size() - 1; i++)
        m_layers.push_back(Layer(true, this->m_shape[i], this->m_shape[i + 1]));

    // adds placeholder neurons.
    m_layers.push_back(Layer(false, this->m_shape[this->m_shape.size() - 1]));
}

float NeuralNet::derivativeOfSigmoid(float output) {
    return output * (1 - output);
}

float NeuralNet::sigmoid(float x) {
    return 1 / (1 + pow(2.71828182846, -x));
}

std::vector<float> NeuralNet::dot(std::vector<float> x1, std::vector<std::vector<float>> x2) {

    LOG(std::cout << "Matrix 1: " << x2.size() << " x " << x2[0].size() << " Matrix 2: " << x1.size() << std::endl);
    std::vector<float> output;

    float t = 0;


    for (auto i = 0; i < x2[0].size(); i++) {
        for (auto a = 0; a < x1.size(); a++) {
            t += x1[a] * x2[a][i];
        }
        output.push_back(sigmoid(t));
        t = 0;
    }



    //https://matrix.reshish.com/multCalculation.php

    return output;
}



#endif // !USER_H