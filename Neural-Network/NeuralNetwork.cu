#pragma once

#include "NeuralNetwork.cuh"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Layer class:
//

NeuralNet::Layer::ANN::ANN(const int deviceNum, int numberOfNeurons, int numberOfNeuronsPrevLayer) {

    cudaSetDevice(deviceNum);

    m_numberNeurons = numberOfNeurons;
    cudaMalloc(&d_activations, sizeof(float) * numberOfNeurons);

    cudaMalloc(&d_weights, sizeof(float) * numberOfNeurons * numberOfNeuronsPrevLayer);

    cudaMalloc(&d_delta, sizeof(float) * numberOfNeurons);
    cudaMalloc(&d_newDelta, sizeof(float) * numberOfNeurons);
    cudaMalloc(&d_error, sizeof(float) * numberOfNeurons);


}


void NeuralNet::Layer::ANN::setActivation(const int deviceNum, std::vector<float> a) {
    cudaSetDevice(deviceNum);
    cudaMemcpy(d_activations, &a[0], m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);
}

float* NeuralNet::Layer::ANN::getActivations(const int deviceNum) {
    cudaSetDevice(deviceNum);

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

    m_layers[0].setActivation(m_deviceNum, input);
}

void NeuralNet::setRandomInput(int64_t seed) {
    cudaSetDevice(m_deviceNum);
    if (m_numberLayers <= 0){
        std::cout << "\033[1;31ERROR:\033[0m In setRandomInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateUniform(gen, m_layers.front().d_activations, m_shape[0]);

    CHECK_FOR_KERNEL_ERRORS;
}

void NeuralNet::init(std::string name, int64_t seed, const float weightDivisor, const float defualtWeight) {
    cudaSetDevice(m_deviceNum);

    cudaStreamCreate(&m_deviceStream);

    srand(seed);

    m_name = name;

    m_layers.clear();

    m_layers.reserve(m_shape.size());
    m_numberLayers = (uint32_t)m_shape.size();

    // Adds placeholder neurons
    m_layers.emplace_back(Layer::ANN(m_deviceNum, m_shape[0]));
    m_totalNumberOfNeurons = m_shape[0];

    for (int i = 1; i < m_shape.size(); i++) {
        m_layers.emplace_back(Layer::ANN(m_deviceNum, m_shape[i], m_shape[i - 1]));
        m_totalNumberOfNeurons += m_shape[i];


        if (defualtWeight != NULL) {
            dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
            dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

            GpuHelperFunc::setAllElemetnsInArrayToOneVal << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[i].d_weights, m_shape[i] * m_shape[i - 1], defualtWeight);

            CHECK_FOR_KERNEL_ERRORS;
        }



    }
    if (defualtWeight == NULL) {

        curandCreateGenerator(&m_cudaRandGen, CURAND_RNG_PSEUDO_XORWOW);

        curandSetPseudoRandomGeneratorSeed(m_cudaRandGen, seed);
    
        random(seed);
    }

    if (weightDivisor != NULL && weightDivisor != 1 && weightDivisor != 0) {

        for (int i = 1; i < m_shape.size(); i++) {

            dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
            dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

            GpuHelperFunc::forEach::constVal::div << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[i].d_weights, m_layers[i].d_weights, m_shape[i] * m_shape[i - 1], weightDivisor);

            CHECK_FOR_KERNEL_ERRORS;

        }
    }
    cublasCreate(&m_feedForwardHandle);

    if (m_activationFunctions.size() - 1 != m_shape.size()) {
        m_activationFunctions.resize(m_shape.size() - 1, "linear");
    }
}


/**
  This function cant accept \
 Please use /
 *
 */
void NeuralNet::save(std::string path) {
    cudaSetDevice(m_deviceNum);

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
         
            saveFile.write((const char*)&m_layers[layerNum].m_bias, sizeof(float));

        }



        saveFile.close();
        std::cout << "Done saving!\n";

    }
    else {
        std::cout << "\033[1;31ERROR\033[0m in save() failed to save model. Caused by: " << m_name << std::endl;
    }


}

void NeuralNet::load(std::string path) {
    cudaSetDevice(m_deviceNum);

    std::cout << "Loading pre trained model...\n";

    std::ifstream loadFile(path, std::ios_base::binary);

    if (loadFile.is_open()) {


        // Get name
        std::string modelName;
        
        std::getline(loadFile, modelName, '\0');

        int sizeOfShape = 0;

        // Get number of layers
        loadFile.read((char*)&sizeOfShape, sizeof(sizeOfShape));
        m_shape.resize(sizeOfShape);
        
        // Get shape of model
        loadFile.read(reinterpret_cast<char*>(&m_shape[0]), sizeOfShape * sizeof(int));
        
        // Initialize without random weights
        init(modelName, 0, 0, 1);


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
            loadFile.read((char*)&m_layers[layer].m_bias, sizeof(float));
        }

        loadFile.close();
    }
    else {
        std::cout << "Could not open file\n";
    }

    std::cout << "\nDone loading pre trained model\n";
}



void NeuralNet::printWeightsAndBias() {
    cudaSetDevice(m_deviceNum);

    printf("[Weights '%s']: \n", m_name.c_str());

    // every col is the weights for one neuron
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray<<<1,1, 0, m_deviceStream >>>(m_layers[layerNum].d_weights, m_shape[layerNum - 1] * m_shape[layerNum]);
        
        CHECK_FOR_KERNEL_ERRORS;
   
        printf("\n");
    }
    
    printf("[Bias '%s']: ", m_name.c_str());
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        printf(" %.6f ", m_layers[layerNum].m_bias);
    }

    printf("\n\n\n");

}

void NeuralNet::printActivations() {
    cudaSetDevice(m_deviceNum);

    printf("[Activations '%s']: \n", m_name.c_str());

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray<<<1,1, 0, m_deviceStream >>>(m_layers[layerNum].d_activations, m_layers[layerNum].m_numberNeurons);

        CHECK_FOR_KERNEL_ERRORS;
    }
    printf("\n\n");
        
}

void NeuralNet::printDeltas() {
    cudaSetDevice(m_deviceNum);

    printf("[Deltas '%s']: \n", m_name.c_str());

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray << <1, 1, 0, m_deviceStream >> > (m_layers[layerNum].d_delta, m_shape[layerNum]);

        CHECK_FOR_KERNEL_ERRORS;

        printf("\n");
    }

}

void NeuralNet::printNewDeltas() {
    cudaSetDevice(m_deviceNum);

    printf("[NewDeltas '%s']: \n", m_name.c_str());

    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray << <1, 1, 0, m_deviceStream >> > (m_layers[layerNum].d_newDelta, m_shape[layerNum]);

        CHECK_FOR_KERNEL_ERRORS;

        printf("\n");
    }

}

void NeuralNet::random(uint64_t seed) {
    cudaSetDevice(m_deviceNum);

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        curandGenerateUniform(m_cudaRandGen, m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons);
        
        CHECK_FOR_KERNEL_ERRORS;
        
        m_layers[layerNum].m_bias = 0;//Random::Default();
    }
}

void NeuralNet::mutate(float mutationStrength) {
    cudaSetDevice(m_deviceNum);

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        Random::MutateArray(m_layers[layerNum].d_weights, &m_cudaRandGen, m_shape[layerNum] * m_shape[layerNum - 1]);

        CHECK_FOR_KERNEL_ERRORS;

        m_layers[layerNum].m_bias *= Random::Default();
    }
}


float NeuralNet::sumOfWeightsAndBias() {
    cudaSetDevice(m_deviceNum);

    float sum = 0;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        GpuHelperFunc::sumOfArray<<<1, 1, 0, m_deviceStream >>>(m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, sum);
        
        CHECK_FOR_KERNEL_ERRORS;

        sum += m_layers[layerNum].m_bias;
    }
    return sum;
}

float* NeuralNet::getOutput() {
    cudaSetDevice(m_deviceNum);

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

    return error / testData.size();
}

void NeuralNet::printOutput() {
    cudaSetDevice(m_deviceNum);

    printf("[Output '%s']: ", m_name.c_str());
    GpuHelperFunc::printArray << <1, 1, 0, m_deviceStream >> > (m_layers[m_numberLayers - 1].d_activations, m_layers[m_numberLayers - 1].m_numberNeurons);
    CHECK_FOR_KERNEL_ERRORS;
}


void NeuralNet::optimizeParametersFeedforward(uint32_t maxGrid, uint32_t maxBlock, uint32_t numberOfTest) {

    uint32_t bestGrid = 1;
    uint32_t bestBlock = 1;

    float progress = 0.0;
    int barWidth = 70;


    setRandomInput(std::rand());

    auto start = std::chrono::high_resolution_clock::now();
    for (auto k = 0; k < numberOfTest; ++k) {
        feedForward(1, 1);
    }
    auto minDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();


    for (uint32_t grid = 1; grid <= maxGrid; ++grid) {
        for (uint32_t block = 2; block <= maxBlock; ++block) {
            
            // progress bar
            progress += 1.0f / ((maxGrid) * (maxBlock - 1));
            std::cout << "[Optimizing Feedforward]: [";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " %\r";
            std::cout.flush();


            //printf("%d, %d\n", grid, block);

            auto start = std::chrono::high_resolution_clock::now();
            for (auto k = 0; k < numberOfTest; ++k) {
                feedForward(grid, block);
            }
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() / numberOfTest;

            if (duration < minDuration) {
                minDuration = duration;
                bestBlock = block;
                bestGrid = grid;
            }


        }

    }
    std::cout.flush();


    m_gridFeedforward = bestGrid;
    m_blockFeedforward = bestBlock;

    printf("\n[Optimized Feedforward]:  (%d, %d)\n", bestGrid, bestBlock);



    
}

void NeuralNet::printShape() {

    printf("[Shape '%s']: ", m_name.c_str());


    for (auto x : m_shape) {
        std::cout << x << std::endl;
    }

}

void NeuralNet::printSize() {
    uint64_t numberOfVaribles = 0;

    // Activation
    for (auto x : m_shape) {
        numberOfVaribles += x;
    }

    // Bias
    numberOfVaribles += m_shape.size();

    // Weights
    for (uint32_t i = 1; i < m_numberLayers; i++) {
        numberOfVaribles += m_shape[i - 1] * m_shape[i];
    }

    std::cout << "Number of parameter in neuralNetwork: " << numberOfVaribles
        << " size in bytes: " << numberOfVaribles * sizeof(float) << "\n"
        << " size in MB: " << (float)numberOfVaribles * sizeof(float) / 1024 / 1024 << "\n"
        << " size in GB: " << (float)numberOfVaribles * sizeof(float) / 1024 / 1024 / 1024 << "\n";
}

float NeuralNet::sumOfDeltas() {

    float sum = 0;

    for (int i = 0; i < m_numberLayers; i++) {
    
        GpuHelperFunc::sumOfArray<<<1, 1 >>>(m_layers[i].d_delta, m_shape[i], sum);
        CHECK_FOR_KERNEL_ERRORS;
    }

}


int NeuralNet::getActivationFuncNum(const int layerNum) {
    // shifting layerNum by -1 because the first layer dont need to be "activated"
    if (m_activationFunctions[layerNum - 1] == "sigmoid") {return 0; }
    else if (m_activationFunctions[layerNum - 1] == "relu") { return 1; }
    else if (m_activationFunctions[layerNum - 1] == "tanh") { return 2; }
    else if (m_activationFunctions[layerNum - 1] == "linear") { return 3; }
    else if (m_activationFunctions[layerNum - 1] == "custom") { return 4; }
    else { std::runtime_error("activation function does not exist"); }
}

#endif // !NEURALNETWORK_CPP