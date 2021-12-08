#pragma once

#include "NeuralNetwork.hpp"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Layer class:
//

NeuralNet::Layer::ANN::ANN(int numberOfNeurons, int numberOfNeuronsPrevLayer, const float defualtWeight) {
    m_numberNeurons = numberOfNeurons;
    m_activation.resize(numberOfNeurons, 0);
    m_bias.resize(numberOfNeurons, 1);

    if (defualtWeight != NULL) {
        m_weights.resize(numberOfNeuronsPrevLayer * numberOfNeurons, defualtWeight);
    }
    else {
        // Random numbers between -1 and 1
        m_weights.resize(numberOfNeuronsPrevLayer * numberOfNeurons);
        for (auto i = 0; i < m_weights.size(); i++) {
            m_weights[i] = Random::Default();
        }
    }
}

float NeuralNet::Layer::ANN::activationFunction(float x) {

    return ACTIVATION_FUNCTION_CPU(x);

}

void NeuralNet::Layer::ANN::writeWeights(std::vector<std::vector<float>>* weight) {

    for (uint32_t i = 0; i < this->m_numberNeurons; i++) {
        weight->emplace_back(m_weights[i]);
    }

}

void NeuralNet::Layer::ANN::setActivation(std::vector<float>* a) {

    for (uint32_t i = 0; i < a->size(); i++) {
        m_activation[i] = (*a)[i];
    }
}

//
// NeuralNet class:
//


void NeuralNet::addLayer(int numberOfNeurons) {
    m_shape.push_back(numberOfNeurons);
}



void NeuralNet::setInput(std::vector<float> input) {
    if (m_numberLayers <= 0) {
        std::cout << "\033[1;31ERROR:\033[0m In setInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    if (input.size() != m_shape[0]) {
        std::cout << "\033[1;33mWARNING: \033[0;0m In setInput() input size not matching networks first layer make sure to call init(). Every activation missing set to zero and extra values deleted. \nInput size: " << input.size() << " first layer size: " << m_shape[0] << ". Caused by: " << m_name << std::endl;
        input.resize(m_shape[0], 1);
    }

    m_layers[0].setActivation(&input);
}

void NeuralNet::setRandomInput() {
    if (m_numberLayers <= 0){
        std::cout << "\033[1;31ERROR:\033[0m In setRandomInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    for (uint32_t i = 0; i < m_layers.front().m_numberNeurons; i++) {
        m_layers.front().m_activation[i] = (static_cast<float> (rand()) / RAND_MAX) * 2 - 1;
    }

}

void NeuralNet::init(std::string name, const float defualtWeight) {

    m_name = name;

    // clear layers if init is already called
    if (m_totalNumberOfNeurons >= m_shape[0]) {
        for (auto i = 0; i < m_shape.size(); i++) {
            if (m_shape[i] != m_layers[i].m_numberNeurons) {
                m_layers.clear();
                break;
            }
        }
        random();
        return;
    }

    m_layers.reserve(m_shape.size());

    // Adds placeholder neurons
    m_layers.emplace_back(Layer::ANN(m_shape[0]));
    m_totalNumberOfNeurons = m_shape[0];

    for (int i = 1; i < m_shape.size() ; i++) {
        m_layers.emplace_back(Layer::ANN(m_shape[i], m_shape[i-1], defualtWeight));
        m_totalNumberOfNeurons += m_shape[i];
    }

    m_numberLayers = (uint32_t)m_shape.size();
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

        // Save name of model
        saveFile.write(m_name.c_str(), m_name.size());
        saveFile.write("\0", sizeof(char)); 

        // Save size of m_shape
        uint32_t sizeOfShape = (uint32_t)m_shape.size();
        saveFile.write(reinterpret_cast<const char*>(&sizeOfShape), sizeof(int));


        // Save shape of model
        saveFile.write(reinterpret_cast<const char*>(&m_shape[0]), m_shape.size() * sizeof(int));

        // Save weights
        for (auto layer : m_layers) {
            for (auto weight : layer.m_weights) {
                saveFile.write((const char*)&weight, sizeof(float));
            }
        }

        // Save bias
        for (auto layer : m_layers) {
            for (auto bias : layer.m_bias) {
                saveFile.write((const char*)&bias, sizeof(float));
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
        std::cout << "Sizeof shape: " << sizeOfShape << std::endl;
        m_shape.resize(sizeOfShape);
        
        // Get shape of model
        loadFile.read(reinterpret_cast<char*>(&m_shape[0]), sizeOfShape * sizeof(int));
        
        // Initialize without random weights
        init(modelName);


        // Load value of weights
        for (uint32_t layer = 0; layer < m_numberLayers; layer++) {

            for (uint32_t weight = 0; weight < m_layers[layer].m_weights.size(); weight++) {

                loadFile.read((char*)&m_layers[layer].m_weights[weight], sizeof(float));

            }


        }

        // Load value of bias
        for (uint32_t layer = 0; layer < m_numberLayers; layer++) {
            for (uint32_t neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
        
                loadFile.read((char*)&m_layers[layer].m_bias[neuron], sizeof(float));
            
            }
        }

        loadFile.close();
    }
    else {
        std::cout << "Could not open file\n";
    }

    std::cout << "\nDone loading pre trained model\n";
}



void NeuralNet::printWeightsAndBias() {

    std::cout << "\n\n\n";


    std::cout << "Weight: \n";

    // every colum is the weights for one neuron
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        for (uint32_t k = 0; k < m_layers[layerNum - 1].m_numberNeurons; k++) {

            for (uint32_t i = 0; i < m_layers[layerNum].m_numberNeurons; i++) {


                std::cout << m_layers[layerNum].m_weights[k * m_layers[layerNum].m_numberNeurons + i] << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    std::cout << "Bias: \n";
    for (auto& layer : m_layers) {

        for (auto& bias : layer.m_bias) {
            std::cout << bias << " | ";
        }
        std::cout << "\n";
    }


    std::cout << "\n\n\n";

}

void NeuralNet::printActivations() {

    std::cout << "Activations: " << std::endl;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        for (auto& activation : m_layers[layerNum].m_activation) {

            std::cout << activation << " ";
        }
        std::cout << "\n";
    }

}

void NeuralNet::random() {
    std::mt19937 gen((uint32_t)time(NULL));
    
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {


        for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_weights.size(); weightNum++) {

            m_layers[layerNum].m_weights[weightNum] = Random::Default();
        }
        
        for (uint32_t neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {


            m_layers[layerNum].m_bias[neuronNum] = Random::Default();

        }

    }
}


float NeuralNet::sumOfWeightsAndBias() {
    float sum{};

    for (auto& layer : m_layers) {

        for (auto& weight : layer.m_weights) {
            sum += float(weight);
        }
    }

    return sum;
}
#endif // !NEURALNETWORK_CPP