#pragma once

#include "NeuralNetwork.cuh"

#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP


//
// Layer class:
//

NeuralNet::Layer::ANN::ANN(uint64_t seed, int numberOfNeurons, int numberOfNeuronsPrevLayer, const float defualtWeight) {

    m_numberNeurons = numberOfNeurons;
    cudaMalloc(&d_activations, sizeof(float) * numberOfNeurons);

    cudaMalloc(&d_weights, sizeof(float) * numberOfNeurons * numberOfNeuronsPrevLayer);
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

void NeuralNet::setRandomInput(int64_t seed) {
    if (m_numberLayers <= 0){
        std::cout << "\033[1;31ERROR:\033[0m In setRandomInput() no valid layers. Number of layers: " << m_numberLayers << " Caused by : " << m_name << std::endl;
        return;
    }
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateUniform(gen, m_layers.front().d_activations, m_shape[0]);

    CHECK_FOR_KERNEL_ERRORS("NeuralNet::setRandomInput()");
}

void NeuralNet::init(std::string name, int64_t seed, const float defualtWeight) {
    

    srand(seed);

    m_name = name;

    {
        //// Checks if existing model is equal to wanted model
        //if (m_numberLayers == m_shape.size()) {

        //    bool equal = true;

        //    for (uint32_t i = 0; i < m_numberLayers; ++i) {

        //        if (m_layers[i].m_numberNeurons != m_shape[i]) {
        //            equal = false;
        //        }
        //    }
        //    if (equal) {
        //        this->random(seed);
        //        return;
        //    }
        //}
    }

    m_layers.clear();

    m_layers.reserve(m_shape.size());
    m_numberLayers = (uint32_t)m_shape.size();

    // Adds placeholder neurons
    m_layers.emplace_back(Layer::ANN(seed, m_shape[0]));
    m_totalNumberOfNeurons = m_shape[0];

    for (int i = 1; i < m_shape.size(); i++) {
        m_layers.emplace_back(Layer::ANN(seed, m_shape[i], m_shape[i - 1], defualtWeight));
        m_totalNumberOfNeurons += m_shape[i];


        if (defualtWeight != NULL) {
            dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
            dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

            GpuHelperFunc::setAllValuesInArrayToOneVal << <DimBlock, DimGrid >> > (m_layers[i].d_weights, m_shape[i] * m_shape[i - 1], defualtWeight);

            CHECK_FOR_KERNEL_ERRORS("NeuralNet::Layer::ANN::ANN()");
        }

    }

    curandCreateGenerator(&m_cudaRandGen, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(m_cudaRandGen, seed);
    
    if (defualtWeight == NULL) {
        random(seed);
    }

    for (int i = 1; i < m_shape.size(); i++) {
     
        dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
        dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

        GpuHelperFunc::multiplyBy2AndSub1 << <DimBlock, DimGrid >> > (m_layers[i].d_weights, m_shape[i] * m_shape[i - 1]);

        CHECK_FOR_KERNEL_ERRORS("NeuralNet::Layer::ANN::ANN()");

    }

    cublasCreate(&m_feedForwardHandle);

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
        init(modelName, 0);


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

    /*std::cout << "Weights:\n";

    // every row is the weights for one neuron
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray<<<1,1>>>(m_layers[layerNum].d_weights, m_shape[layerNum - 1] * m_shape[layerNum]);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::printWeightsAndBias()");
   
        printf("\n");
    }
    */
    std::cout << "Bias:\n";
    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        printf(" %.6f ", m_layers[layerNum].m_bias);
    }

    std::cout << "\n\n\n";

}

void NeuralNet::printActivations() {

    printf("Activations:\n");
    
    for (uint32_t layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::printArray<<<1,1>>>(m_layers[layerNum].d_activations, m_layers[layerNum].m_numberNeurons);

        CHECK_FOR_KERNEL_ERRORS("NeuralNet::printActivations()");
    }
    printf("\n\n");
        
}

void NeuralNet::random(uint64_t seed) {
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        curandGenerateUniform(m_cudaRandGen, m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::random()");
        
       m_layers[layerNum].m_bias = Random::Default();
    }
}

void NeuralNet::mutate(float mutationStrength) {

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        Random::MutateArray(m_layers[layerNum].d_weights, &m_cudaRandGen, m_shape[layerNum] * m_shape[layerNum - 1]);

        CHECK_FOR_KERNEL_ERRORS("NeuralNet::printActivations()");

        m_layers[layerNum].m_bias *= Random::Default();
    }
}


float NeuralNet::sumOfWeightsAndBias() {
    float sum = 0;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        GpuHelperFunc::sumOfArray<<<1, 1>>>(m_layers[layerNum].d_weights, m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons, sum);
        
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::sumOfWeightsAndBias()");

        sum += m_layers[layerNum].m_bias;
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

    printf("[Output '%s']: ", m_name.c_str());
    GpuHelperFunc::printArray << <1, 1 >> > (m_layers[m_numberLayers - 1].d_activations, m_layers[m_numberLayers - 1].m_numberNeurons);
    CHECK_FOR_KERNEL_ERRORS("NeuralNet::printOutput");
}


void NeuralNet::optimizeParametersFeedforward(uint32_t maxGrid, uint32_t maxBlock, uint32_t numberOfTest) {

    uint32_t bestGrid = 1;
    uint32_t bestBlock = 1;

    float progress = 0.0;
    int barWidth = 70;


    setRandomInput(std::rand());

    auto start = std::chrono::high_resolution_clock::now();
    feedForward(1, 1);
    auto minDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();


    for (uint32_t grid = 1; grid <= maxGrid; ++grid) {
        for (uint32_t block = 2; block <= maxBlock; ++block) {
            
            // progress bar
            progress += 1.0f / ((maxGrid) * (maxBlock - 1));
            std::cout << "[Optimizing Feedforward Params]: [";
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
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

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

    printf("\n[Optimized Feedforward Params]: (%d, %d)\n", bestGrid, bestBlock);



    
}

void NeuralNet::printShape() {

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

#endif // !NEURALNETWORK_CPP