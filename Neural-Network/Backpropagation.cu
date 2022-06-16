#pragma once

#include "NeuralNetwork.cuh"


#ifndef BACKPROPAGATION_CU
#define BACKPROPAGATION_CU



__global__ void computeDelta(float* delta, float* newDelta, const float* error, const float* activations, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {

        float x = activations[id];
        float deriv = 0;

        // sigmoid
        if (functionNum == 0) {
            deriv = x * (1 - x);
        }
        // relu
        else if (functionNum == 1) {
            deriv = x > 0 ? 1 : 0;
        }
        // tanh
        else if (functionNum == 2) {
            deriv = expf(2 * x) + 1;
            deriv = (4 / deriv) - (4 / (deriv * deriv));
        }
        // linear
        else if (functionNum == 3) {
            deriv = 1;
        }
        // custom
        else if (functionNum == 4) {
            deriv = (2 / (expf(-x) + 1)) - 1;
        }
        newDelta[id] = error[id] * deriv;
        delta[id] += newDelta[id];
    }
}


__global__ void weightUpdate(const float* delta, const float* prevLayerActivations, float* weights, const float learningRate, const int prevSize, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

   for (; id < prevSize; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
       for (int j = 0; j < size; j++) {
           weights[id * size + j] -= learningRate * delta[j] * prevLayerActivations[id];
       }
   }
}

void NeuralNet::backpropagation(const std::vector<std::vector<float>> dataset, const std::vector<std::vector<float>> correctOutput,
    const float updateWeightsAfterEveryBackPass,
    int batchSize,
    const bool randomBatching,
    const bool averageOutDeltas) {

    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

    float* d_expectedOutput = NULL;

    cudaMalloc(&d_expectedOutput, sizeof(float) * m_shape.back());

    int datasetIndex = 0;

    if (batchSize == NULL) {
        batchSize = dataset.size();
    }

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

        if (randomBatching) {
            datasetIndex = std::rand() % dataset.size();
        }
        else {
            datasetIndex = batchIndex;
        }

        setInput(dataset[datasetIndex]);
        feedForward();

        cudaMemcpy(d_expectedOutput, correctOutput[datasetIndex].data(), m_shape.back() * sizeof(float), cudaMemcpyHostToDevice);
        CHECK_FOR_KERNEL_ERRORS;

        GpuHelperFunc::forEach::sub <<<1, 1, 0, m_deviceStream >> > (m_layers.back().d_error, m_layers.back().d_activations, d_expectedOutput, m_shape.back());
        CHECK_FOR_KERNEL_ERRORS;

        int funcNum = getActivationFuncNum(m_numberLayers - 1);
        computeDelta << <DimGrid, DimBlock, 0, m_deviceStream >> > (
            m_layers.back().d_delta, 
            m_layers.back().d_newDelta,  
            m_layers.back().d_activations, 
            m_layers.back().d_error,
            funcNum, 
            m_shape.back()
        );

        CHECK_FOR_KERNEL_ERRORS;

        for (int layerNum = m_numberLayers - 2; layerNum > 0; layerNum--) {

            GpuHelperFunc::cublasCompute(
                m_feedForwardHandle,
                m_layers[layerNum + 1].d_newDelta,
                m_layers[layerNum + 1].d_weights,
                m_layers[layerNum].d_error,
                m_shape[layerNum],
                1,
                m_shape[layerNum + 1],
                m_deviceNum
            );
            CHECK_FOR_KERNEL_ERRORS;

            int funcNum = getActivationFuncNum(layerNum);
            computeDelta <<<DimGrid, DimBlock, 0, m_deviceStream>>>(
                m_layers[layerNum].d_delta, 
                m_layers[layerNum].d_newDelta,
                m_layers[layerNum].d_error, 
                m_layers[layerNum].d_activations, 
                funcNum, 
                m_shape[layerNum]
            );

            CHECK_FOR_KERNEL_ERRORS;
        }

        if (updateWeightsAfterEveryBackPass != NULL) {
            updateWeights(updateWeightsAfterEveryBackPass);
            clearDelta();

        }

    }
    if (averageOutDeltas) {
        for (int layerNum = m_numberLayers - 1; layerNum > 0; layerNum--) {
            GpuHelperFunc::forEach::constVal::div << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_delta, m_layers[layerNum].d_delta, (float)batchSize, m_shape[layerNum]);
            CHECK_FOR_KERNEL_ERRORS;
        }
    }


    return;
}


void NeuralNet::updateWeights(float learning_rate) {

    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

    for (int layerNum = m_numberLayers - 1; layerNum > 0; layerNum--) {
        weightUpdate << <DimGrid, DimBlock, 0, m_deviceStream >> > (
            m_layers[layerNum].d_delta,
            m_layers[layerNum - 1].d_activations,
            m_layers[layerNum].d_weights,
            learning_rate,
            m_shape[layerNum - 1],
            m_shape[layerNum]
            );
        CHECK_FOR_KERNEL_ERRORS;
    
    }

    return;
}

void NeuralNet::clearDelta() {
    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);

    for (int layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::setAllElemetnsInArrayToOneVal << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_delta,    m_shape[layerNum], 0);
        CHECK_FOR_KERNEL_ERRORS;
    }

    return;
}

#endif //!BACKPROPAGATION_CU