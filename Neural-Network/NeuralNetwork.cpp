#pragma once

#include "NeuralNetwork.hpp"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Neuron class:
//
#define RANGE 0.05f
NeuralNet::Layer::ANN::Neuron::Neuron(std::mt19937* gen, int numberOfNeuronsPrevLayer, const float defualtValue) {

    if (defualtValue != NULL) {
        m_weights.resize(numberOfNeuronsPrevLayer, defualtValue);
    }
    else {
        // Random numbers between -1 and 1
                                             // 2.0f / float(gen.max())
        m_weights.resize(numberOfNeuronsPrevLayer, 4.656612873e-10F);
        for (auto i = 0; i < numberOfNeuronsPrevLayer; i++) {
            m_weights[i] *= float((*gen)()) * RANGE;
            m_weights[i] -= RANGE;
        }
    }
}

float NeuralNet::Layer::ANN::Neuron::activationFunction(float x) {
   
    return x;

}

/*
-0.548858 0.00775933 0.603525
0.410437  0.681789   0.260143
-0.60173  -0.0700927 -0.480505
0.128276  0.22018   -0.205762
-0.317934 0.930447  -0.291043 


-0.548858 0.410437 -0.60173 0.128276 -0.317934
0.00775933 0.681789 -0.0700927 0.22018 0.930447
0.603525 0.260143 -0.480505 -0.205762 -0.291043
*/

//
// Layer class:
//


NeuralNet::Layer::ANN::ANN(std::mt19937* gen, int numberOfNeurons, int numberOfNeuronsPrevLayer, const float defualtWeight) {
    m_numberNeurons = numberOfNeurons;
    m_neurons.reserve(numberOfNeurons);
    for (uint32_t i = 0; i < m_numberNeurons; i++) {
        m_neurons.emplace_back(Neuron(gen, numberOfNeuronsPrevLayer, defualtWeight));
    }
}

void NeuralNet::Layer::ANN::getWeights(std::vector<std::vector<float>>* weight) {

    for (uint32_t i = 0; i < this->m_numberNeurons; i++) {
        weight->emplace_back(m_neurons[i].m_weights);
    }

}


std::vector<float> NeuralNet::Layer::ANN::getBias() {
    std::vector<float> a; 
    a.reserve(m_numberNeurons);

    for (uint32_t i = 0; i < m_numberNeurons; i++) {
        a.emplace_back(m_neurons[i].m_bias);

    }
    return a;
}

std::vector<float> NeuralNet::Layer::ANN::getActivations() {

    std::vector<float> out;
    out.reserve(m_numberNeurons);
    
    for (uint32_t i = 0; i < m_numberNeurons; i++) {
        out.emplace_back(m_neurons[i].m_activation);
    }

    return out;
}
void NeuralNet::Layer::ANN::setActivation(std::vector<float>* a) {

    for (uint32_t i = 0; i < a->size(); i++) {
        m_neurons[i].m_activation = (*a)[i];
    }
}


void NeuralNet::Layer::ANN::getWeights1D(std::vector<float> *writeArray) {
    //std::cout << "\t totalsize: " << totalsize << " " << m_neurons.front().m_weights.size() << std::endl;

    writeArray->clear();
    writeArray->reserve(m_numberNeurons * m_neurons.front().m_weights.size());

    
    for (uint32_t x = 0; x < m_numberNeurons; x++) {
        for (size_t y = 0; y < m_neurons.front().m_weights.size(); y++) {
            writeArray->emplace_back(m_neurons[x].m_weights[y]);
        }
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

void NeuralNet::setRandomInput() {
    
    for (uint32_t i = 0; i < m_layers.front().m_numberNeurons; i++) {
        m_layers.front().m_neurons[i].m_activation = (static_cast<float> (rand()) / RAND_MAX) * 2 - 1;
    }

}

void NeuralNet::init(std::string name, const float defualtWeight) {

    // Random number generator
    std::mt19937 gen(static_cast<unsigned int>( std::chrono::system_clock::now().time_since_epoch().count()));

    m_name = name;

    // clear layers if init is already called
    m_layers.clear();

    // Reserve memory
    m_layers.reserve(m_shape.size());


    // Adds placeholder neurons
    m_layers.emplace_back(Layer::ANN(&gen, m_shape[0]));
    
    m_totalNumberOfNeurons = m_shape[0];

    for (int i = 1; i < m_shape.size() ; i++) {
        m_layers.emplace_back(Layer::ANN(&gen, m_shape[i], m_shape[i-1], defualtWeight));
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

        // Save name
        saveFile.write(m_name.c_str(), m_name.size());
        saveFile.write("\0", sizeof(char)); 

        // Save size of m_shape
        uint32_t sizeOfShape = (uint32_t)m_shape.size();
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
        std::cout << "Sizeof shape: " << sizeOfShape << std::endl;
        m_shape.resize(sizeOfShape);
        
        // Get shape of network
        loadFile.read(reinterpret_cast<char*>(&m_shape[0]), sizeOfShape * sizeof(int));
        
        // Initialize without random weights
        init(modelName);


        // Load value of weights
        for (uint32_t layer = 0; layer < m_numberLayers; layer++) {

            for (uint32_t neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
            
                for (uint32_t weight = 0; weight < m_layers[layer].m_neurons[neuron].m_weights.size(); weight++) {
                    loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_weights[weight], sizeof(float));
                
                }
            
            }
        
        }

        // Load value of bias
        for (uint32_t layer = 0; layer < m_numberLayers; layer++) {
            for (uint32_t neuron = 0; neuron < m_layers[layer].m_numberNeurons; neuron++) {
        
                loadFile.read((char*)&m_layers[layer].m_neurons[neuron].m_bias, sizeof(float));
            
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

    // every colum is weights for one neuron
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        for (uint32_t k = 0; k < m_layers[layerNum].m_neurons.front().m_weights.size(); k++) {

            for (uint32_t i = 0; i < m_layers[layerNum].m_numberNeurons; i++) {


                std::cout << m_layers[layerNum].m_neurons[i].m_weights[k] << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    std::cout << "Bias: \n";
    for (auto& layer : m_layers) {

        for (auto& neuron : layer.m_neurons) {
            std::cout << neuron.m_bias << " | ";
        }
        std::cout << "\n";
    }


    std::cout << "\n\n\n";

}

void NeuralNet::printActivations() {

    std::cout << "Activations: " << std::endl;

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {

        for (auto& neuron : m_layers[layerNum].m_neurons) {

            std::cout << neuron.m_activation << " |";
        }
        std::cout << "\n";
    }

}

void NeuralNet::random() {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::mt19937 gen((uint32_t)time(NULL));
    
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {

        for (uint32_t neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {


            // Choses numberOfWeights / 3 random weights and takes the average
            for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {

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