#pragma once

#include "NeuralNetwork.cuh"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Layer class:
//


NeuralNet::Layer::ANN::ANN(int numberOfNeurons, int numberOfNeuronsPrevLayer, const float defualtWeight) {

    m_numberNeurons = numberOfNeurons;
    cudaMalloc(&d_activations, sizeof(float) * numberOfNeurons);
    

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    

    cudaMalloc(&d_weights, sizeof(float) * numberOfNeurons * numberOfNeuronsPrevLayer);
    if (defualtWeight != NULL) {
        GpuHelperFunc::setAllValuesInArrayToOneVal << <DimBlock, DimGrid >> > (d_weights, numberOfNeurons * numberOfNeuronsPrevLayer, defualtWeight);
        cudaDeviceSynchronize();
    }
    else {
        // Random numbers between -1 and 1
        Random::ArrayGpu << < DimBlock, DimGrid >> > (d_weights, numberOfNeurons * numberOfNeuronsPrevLayer, Random::d_x, Random::d_y, Random::d_z);
        cudaDeviceSynchronize();
    }

}


float NeuralNet::Layer::ANN::activationFunction(float x) {
    return ACTIVATION_FUNCTION_CPU(x);
}

void NeuralNet::Layer::ANN::setActivation(float* a) {
    cudaMemcpy(d_activations, a, m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);
}

//
// NeuralNet class:
//


void NeuralNet::addLayer(int numberOfNeurons) {
    m_shape.push_back(numberOfNeurons);
}

void NeuralNet::setInput(float* input, const size_t size) {
    if (m_numberLayers <= 0) {
        std::cout << "\033[1;31 ERROR:\033[0m In setInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    if (size != m_shape[0]) {
        std::cout << "\033[1;31 ERROR:\033[0m In setInput() input size not matching networks first layer make sure to call init(). Caused by: " << m_name << std::endl;
        return;
    }

    m_layers[0].setActivation(input);
}

void NeuralNet::setRandomInput() {
    if (m_numberLayers <= 0){
        std::cout << "\033[1;31ERROR:\033[0m In setRandomInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    Random::ArrayGpu << < DimBlock, DimGrid >> > (m_layers.front().d_activations, m_layers.front().m_numberNeurons, Random::d_x, Random::d_y, Random::d_z);
    cudaDeviceSynchronize();


}

void NeuralNet::init(std::string name, const float defualtWeight) {

    m_name = name;

    if (m_totalNumberOfNeurons >= m_shape[0]) {
        for (auto i = 0; i < m_shape.size(); i++) {
            if (m_shape[i] != m_layers[i].m_numberNeurons) {
                m_layers.clear();
                break;
            }
        }
        this->random();
        return;
    }

    m_layers.reserve(m_shape.size());
    m_numberLayers = (uint32_t)m_shape.size();

    // Adds placeholder neurons
    m_layers.emplace_back(Layer::ANN(m_shape[0]));
    m_totalNumberOfNeurons = m_shape[0];

    for (int i = 1; i < m_shape.size() ; i++) {
        m_layers.emplace_back(Layer::ANN(m_shape[i], m_shape[i-1], defualtWeight));
        m_totalNumberOfNeurons += m_shape[i];
    }

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
        for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
            for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons; weightNum++) {
                saveFile.write((const char*)&m_layers[layerNum].d_weights[weightNum], sizeof(float));
            }
        }

        // Save bias
        for (auto& layer : m_layers) {
         
            saveFile.write((const char*)&layer.d_bias, sizeof(float));

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
        for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {

            for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons; weightNum++) {

                loadFile.read((char*)&m_layers[layerNum].d_weights[weightNum], sizeof(float));

            }


        }

        // Load value of bias
        for (uint32_t layer = 0; layer < m_numberLayers; layer++) {
            loadFile.read((char*)&m_layers[layer].d_bias, sizeof(float));
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

        GpuHelperFunc::printArray << <1, 1 >> > (m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons, m_layers[layerNum].m_numberNeurons);
        cudaDeviceSynchronize();
    }


    std::cout << "Bias: \n";
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        printf("%d ", m_layers[layerNum].d_bias);
    }


    std::cout << "\n\n\n";

}

void NeuralNet::printActivations() {

    std::cout << "Activations: " << std::endl;

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray << <1, 1 >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_numberNeurons, m_layers[layerNum].m_numberNeurons);
        cudaDeviceSynchronize();
    }

}

void NeuralNet::random() {
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        Random::ArrayGpu << < DimBlock, DimGrid >> > (m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, Random::d_x, Random::d_y, Random::d_z);
        
        cudaDeviceSynchronize();

        m_layers[layerNum].d_bias = Random::Default();
    }
}


float NeuralNet::sumOfWeightsAndBias() {
    float sum = 0;

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {

        GpuHelperFunc::sumOfArray<<<1, 1>>>(m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, sum);

    }
    printf("Sum isnt right\n");
    return sum;
}

float* NeuralNet::getOutput() {

    float* output;
    output = (float*)malloc(m_layers.back().m_numberNeurons * sizeof(float));
    cudaMemcpy(output, m_layers.back().d_activations, m_layers.back().m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

#endif // !NEURALNETWORK_CPP