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
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::Layer::ANN::ANN()");
        
        cudaDeviceSynchronize();
    }
    else {
        // Random numbers between -1 and 1
        Random::ArrayGpu << < DimBlock, DimGrid >> > (d_weights, numberOfNeurons * numberOfNeuronsPrevLayer, Random::offset + std::rand());
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::Layer::ANN::ANN()");
        
        cudaDeviceSynchronize();
    }

}


void NeuralNet::Layer::ANN::setActivation(std::vector<float> a) {
    cudaMemcpy(d_activations, &a[0], m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);
}

float* NeuralNet::Layer::ANN::getActivations() {

    float* out;
    out = (float*)malloc(m_numberNeurons * sizeof(float));
    cudaMemcpy(out, d_activations, m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);
    return out;
}

//
// NeuralNet class:
//


void NeuralNet::addLayer(int numberOfNeurons) {
    m_shape.push_back(numberOfNeurons);
}

void NeuralNet::setInput(std::vector<float> input) {
    if (m_numberLayers <= 0) {
        std::cout << "\033[1;31 ERROR:\033[0m In setInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    if (input.size() != m_shape[0]) {
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
    
    Random::ArrayGpu << < DimBlock, DimGrid >> > (m_layers.front().d_activations, m_layers.front().m_numberNeurons, Random::offset + std::rand());

    CHECK_FOR_KERNEL_ERRORS("NeuralNet::setRandomInput()");

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
        for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

            float* tmp;
            tmp = (float*)malloc(m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
            cudaMemcpy(tmp, m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

            for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons; weightNum++) {
                saveFile.write((const char*)&tmp[weightNum], sizeof(float));
            }
            free(tmp);
        }

        // Save bias
        for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
         
            saveFile.write((const char*)&m_layers[layerNum].d_bias, sizeof(float));

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
        for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
            
            float* tmp;
            tmp = (float*)malloc(m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
            for (uint32_t weightNum = 0; weightNum < m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons; weightNum++) {

                loadFile.read((char*)&tmp[weightNum], sizeof(float));

            }
            cudaMemcpy(m_layers[layerNum].d_weights, tmp, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);


        }

        // Load value of bias
        for (uint32_t layer = 1; layer < m_numberLayers; layer++) {
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

    std::cout << "Weights: \n";

    // every colum is the weights for one neuron
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray<<<1, 1>>>(m_layers[layerNum].d_weights, m_layers[layerNum - 1].m_numberNeurons * m_layers[layerNum].m_numberNeurons);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::printWeightsAndBias()");
        cudaDeviceSynchronize();
    
        printf("\n");
    }


    std::cout << "Bias: \n";
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        printf(" %.6f ", m_layers[layerNum].d_bias);
    }


    std::cout << "\n\n\n";

}

void NeuralNet::printActivations() {

    std::cout << "Activations: " << std::endl;
    printf("n: %d", m_numberLayers);
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray << <1, 1 >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_numberNeurons);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::printActivations()");
        
        cudaDeviceSynchronize();
    }

}

void NeuralNet::random() {
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        Random::ArrayGpu << < DimBlock, DimGrid >> > (m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, Random::offset + rand());
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::random()");
        
        cudaDeviceSynchronize();

        m_layers[layerNum].d_bias = Random::Default();
    }
}

void NeuralNet::mutate(float mutationStrength) {
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        Random::MutateArrayGpu << < DimBlock, DimGrid >> > (m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, Random::offset + rand());

        CHECK_FOR_KERNEL_ERRORS("NeuralNet::mutate()");

        cudaDeviceSynchronize();

        m_layers[layerNum].d_bias *= Random::Default();
    }
}


float NeuralNet::sumOfWeightsAndBias() {
    float sum = 0;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        GpuHelperFunc::sumOfArray<<<1, 1>>>(m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, sum);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::sumOfWeightsAndBias()");

        sum += m_layers[layerNum].d_bias;
    }
    return sum;
}

float* NeuralNet::getOutput() {

    float* output;
    output = (float*)malloc(m_layers.back().m_numberNeurons * sizeof(float));
    cudaMemcpy(output, m_layers.back().d_activations, m_layers.back().m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

float NeuralNet::MAELossFunction(float* output, std::vector<float> target) {
    float error = 0; 
    
    for (size_t i = 0; i < target.size(); i++) {
        error += std::abs(output[i] - target[i]);
    }

    return std::abs(error) / target.size();

}

float NeuralNet::MSELossFunction(float* output, std::vector<float> target){
    float error = 0;

    for (size_t i = 0; i < target.size(); i++) {
        error += std::powf(output[i] - target[i], 2);
    }

    return std::abs(error) / target.size();
}

float NeuralNet::LossFunction(float* output, std::vector<float> target) {

    return MAELossFunction(output, target);

}

float NeuralNet::performTest(std::vector<std::vector<float>> testData, std::vector<std::vector<float>> expectedOutput ) {

    float error = 0;

    for (size_t i = 0; i < testData.size(); i++) {

        setInput(testData[i]);
        
        error += LossFunction(feedForward(), expectedOutput[i]);
    }

    return error;
}

void NeuralNet::printOutput() {

    std::vector<float> output;
    output.reserve(m_shape.back() * sizeof(float));

    memcpy(&output[0], getOutput(), m_shape.back() * sizeof(float));


    printf("Output: ");
    for (auto i = 0; i < output.size(); i++) std::cout << output[i] << " | ";
    printf("\n");
}

#endif // !NEURALNETWORK_CPP