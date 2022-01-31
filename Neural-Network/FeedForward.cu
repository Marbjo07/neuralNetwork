#pragma once

#include "NeuralNetwork.cuh"


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP


/*
#define SHARED_MEMORY_SIZE  1024


__global__ void matrixMul(const float* activations, const float* weights, const float bias, float* output, const int currentSize, const int prevSize) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int size = floor((float)currentSize / MAX);

    __shared__ float cachedActivations[SHARED_MEMORY_SIZE];

    int idx = threadIdx.x;
    while (idx < SHARED_MEMORY_SIZE && idx < prevSize) {
        cachedActivations[idx] = activations[idx];
        idx += blockDim.x;
    }

    //__syncthreads();

    // Distrubute the remaining activations to the last few threads
    if (id == MAX) {
        size += currentSize % MAX;
    }

    id *= size;
    int end = id + size;

    float tmp = 0;
    int i = 0;

    for (; id < end; ++id) {
        tmp = 0;

        for (i = 0; i < prevSize && i < SHARED_MEMORY_SIZE; ++i) {
            tmp += cachedActivations[i] * weights[prevSize * id + i];
        }
        for (; i < prevSize; i++) {
            tmp += activations[i] * weights[prevSize * id + i];
        }

        output[id] = ACTIVATION_FUNCTION_GPU(tmp) + bias;
    }
}*/

__global__ void activationFunction(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


#pragma unroll
    for (; id < size; id += blockDim.x * blockDim.y) {

        float x = activations[id];

        switch (functionNum) {

        // sigmoid
        case 0:
            activations[id] = 1 / expf(x);
            break;
        // relu
        case 1:
            activations[id] = fmaxf(x, 0);
            break;
        // tanh
        case 2:
            activations[id] = tanh(x);
            break;
        // none
        case 3:
            break;
        // custom
        case 4:
            activations[id] = 1 / (1 - x + x * x);
            break;
        }


        activations[id] += bias;
    }
}

__global__ void softMax_gpu(float* activations, const int size) { 

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
   
    __shared__ float sum;


    if (id == 0) {
#pragma unroll
        for (int i = 0; i < size; i++) {
            sum += expf(activations[i]);
        }
    }

    __syncthreads();

    for (; id < size; id += blockDim.x * blockDim.y) {
        activations[id] = expf(activations[id]) / sum;
    }

}

void NeuralNet::softMax() {
    
    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    GpuHelperFunc::usePrintArrayFromCppFile(m_layers.back().d_activations, m_shape.back());
    softMax_gpu << <DimBlock, DimGrid >> > (m_layers.back().d_activations, m_shape.back());
    CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
    GpuHelperFunc::usePrintArrayFromCppFile(m_layers.back().d_activations, m_shape.back());
}

float* NeuralNet::feedForward(uint32_t gridSize, uint32_t blockSize) {


    if (gridSize == NULL || gridSize <= 0) {
        gridSize = m_gridFeedforward;
    }

    if (blockSize == NULL || blockSize <= 0) {
        blockSize = m_blockFeedforward;
    }

    dim3 DimGrid(gridSize, gridSize, 1);
    dim3 DimBlock(blockSize, blockSize, 1);

    float alpha = 1;
    float beta = 0;


    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::cublasCompute(
            m_feedForwardHandle,
            m_layers[layerNum - 1].d_activations,
            m_layers[layerNum].d_weights,
            m_layers[layerNum].d_activations,
            m_shape[layerNum],
            1,
            m_shape[layerNum - 1]
        );

        /*
        matrixMul << <DimBlock, DimGrid >> > (
            m_layers[layerNum - 1].d_activations,
            m_layers[layerNum].d_weights,
            m_layers[layerNum].m_bias,
            m_layers[layerNum].d_activations,
            m_layers[layerNum].m_numberNeurons,
            m_layers[layerNum - 1].m_numberNeurons);
        */


        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");

        // shifting layerNum by -1 because the first layer dont need to be "activated"
        int functionNum;
        if (m_activationFunctions[layerNum - 1] == "sigmoid") { functionNum = 0; }
        else if (m_activationFunctions[layerNum - 1] == "relu") { functionNum = 1; }
        else if (m_activationFunctions[layerNum - 1] == "tanh") { functionNum = 2; }
        else if (m_activationFunctions[layerNum - 1] == "none") { functionNum = 3; }
        else if (m_activationFunctions[layerNum - 1] == "custom") { functionNum = 4; }
        else { std::runtime_error("activation function does not exist"); }



        activationFunction << <DimGrid, DimBlock >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");



    }

 

    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/