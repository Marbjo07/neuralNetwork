#pragma once

#include "NeuralNetwork.hpp"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Neuron class:
//

NeuralNet::Layer::Neuron::Neuron(int numberOfNeuronsPrevLayer, int defualtValue) {

    if (defualtValue != NULL) {
        m_weights.resize(numberOfNeuronsPrevLayer, defualtValue);
    }
    else {
        for (auto i = 0; i < numberOfNeuronsPrevLayer; i++) {
            m_weights.push_back(static_cast <float> (rand()) / RAND_MAX * 2 - 1);
        }
    }
}

void NeuralNet::Layer::Neuron::mutateWeightAndBias(float mutationStrength) {

    for (auto i = 0; i < m_weights.size(); i++) {
        if (m_weights[i] < 10) {
            m_weights[i] += (static_cast <float> (rand()) / RAND_MAX * 2 - 1) * mutationStrength;
        }
        else {
            m_weights[i] = 1;
        }
    }
    if (m_bias < 10) {
        m_bias += (static_cast <float> (rand()) / RAND_MAX * 2 - 1) * mutationStrength;
    }
    else {
        m_bias = 1;
    }
}


float NeuralNet::Layer::Neuron::activationFunction(float x) {

    // sigmoid but its streched beacuse then its easyer for the ai to keep data
    
//return (10 / (1 + pow(1.6, -x))) + m_bias - 5;
    return x / (1 + abs(x)) + m_bias;

}

//
// Layer class:
//


NeuralNet::Layer::Layer(int numberOfNeurons, int numberOfNeuronsPrevLayer, int defualtWeight) {
    m_numberNeurons = numberOfNeurons;

    for (auto i = 0; i < m_numberNeurons; i++) {
        m_neurons.push_back(Neuron(numberOfNeuronsPrevLayer, defualtWeight));
    }
}

void NeuralNet::Layer::getWeights(std::vector<std::vector<float>>* weight) {

    for (auto i = 0; i < this->m_numberNeurons; i++) {
        weight->emplace_back(m_neurons[i].m_weights);
    }

}


std::vector<float> NeuralNet::Layer::getBias() {
    std::vector<float> a; 

    for (auto i = 0; i < m_numberNeurons; i++) {
        a.emplace_back(m_neurons[i].m_bias);

    }
    return a;
}

std::vector<float> NeuralNet::Layer::getActivation() {

    std::vector<float> out;
    out.reserve(m_numberNeurons);
    for (auto i = 0; i < m_numberNeurons; i++) {
        out.emplace_back(m_neurons[i].m_activation);
    }

    return out;
}
void NeuralNet::Layer::setActivation(std::vector<float>* a) {

    for (auto i = 0; i < a->size(); i++) {
        m_neurons[i].m_activation = (*a)[i];
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



void NeuralNet::setInput(std::vector<float> input) {
    if (input.size() != m_shape[0]) {
        std::cout << "\033[1;33mWARNING: \033[0;0m In setInput() input size not matching networks first layer. Unexpected behavior may occur. Input size: " << input.size() << " first layer size: " << m_shape[0] << ". Caused by: " << m_name << std::endl;
        input.resize(m_shape[0], 1);
    }

    m_layers[0].setActivation(&input);
}

void NeuralNet::init(std::string name, int defualtWeight) {

    m_name = name;

    // clear layers if init is already called
    m_layers.clear();
    
    // Reserve memory
    m_layers.reserve(m_shape.size());

    // Adds placeholder neurons
    m_layers.push_back(Layer(m_shape[0]));
    
    m_totalNumberOfNeurons += m_shape[0];

    for (int i = 1; i < m_shape.size() ; i++) {
        m_layers.push_back(Layer(m_shape[i], m_shape[i-1], defualtWeight));
        m_totalNumberOfNeurons += m_shape[i];
    }

    m_numberLayers = m_shape.size();
}

/**
  This function cant accept \
 Please use /
 *
 */
void NeuralNet::save(std::string path) {

    std::cout << "Saving... \n";

    std::ofstream saveFile(path, std::ios_base::binary);

    if (saveFile.is_open()) {

        // Save name
        saveFile.write(m_name.c_str(), m_name.size());

        // Save size of m_shape
        int sizeOfShape = m_shape.size();
        saveFile.write(reinterpret_cast<const char*>(&sizeOfShape), sizeof(int));


        // Save shape of model
        saveFile.write(reinterpret_cast<const char*>(&m_shape[0]), m_shape.size() * sizeof(int));

        // Save weights
        for (auto layer : m_layers) {
            for (auto neuron : layer.m_neurons) {
                for (auto weight : neuron.m_weights) {
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
        std::cout << "Done saving!\n";

    }
    else {
        std::cout << "\033[1;31ERROR\033[0m in save() failed to save model. Caused by: " << m_name << std::endl;
    }


}

void NeuralNet::load(std::string path) {
    std::cout << "Loading pre trained model...\n";

    std::ifstream loadFile(path, std::ios_base::binary);

    if (loadFile.is_open()) {


        // Get name
        std::string modelName;
        std::getline(loadFile, modelName, '\0');

        int sizeOfShape;

        // Get number of layers
        loadFile.read((char*)&sizeOfShape, sizeof(sizeOfShape));

        m_shape.resize(sizeOfShape);
        
        // Get shape of network
        loadFile.read(reinterpret_cast<char*>(&m_shape[0]), sizeOfShape * sizeof(int));
        
        // Initialize without random weights
        init(modelName, true);


        // Load value of weights
        for (auto layer = 0; layer < m_numberLayers; layer++) {
            for (auto neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
                for (auto weight = 0; weight < m_layers[layer].m_neurons[neuron].m_weights.size(); weight++) {
                    loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_weights[weight], sizeof(float));
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

    for (auto layerNum = 0; layerNum < m_numberLayers; layerNum++) {

        for (auto neuron : m_layers[layerNum].m_neurons) {

            for (auto weight : neuron.m_weights) {
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


void NeuralNet::random() {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::mt19937 gen(time(NULL));
    
    for (auto layerNum = 0; layerNum < m_numberLayers; layerNum++) {

        for (auto neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {


            // Choses numberOfWeights / 3 random weights and takes the average
            for (auto weightNum = 0; weightNum < m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {

                m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum] = static_cast<float>(gen()) / gen.max();
            }

            m_layers[layerNum].m_neurons[neuronNum].m_bias = static_cast<float>(gen()) / gen.max();


        }

    }
    std::cout << "Duration in miliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;

}


float NeuralNet::sumOfWeightsAndBias() {
    float sum{};

    for (auto &layer : m_layers) {

        for (auto &neuron : layer.m_neurons) {
            
            for (auto &weight : neuron.m_weights) {
                sum += float(weight);
            }
        }
    }

    return sum;
}

#endif // !NEURALNETWORK_CPP