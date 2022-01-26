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

__global__ void actiavtionFunction(float* activations, float bias,  const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


#pragma unroll
    for (; id < size; id += blockDim.x * blockDim.y) {
        activations[id] = ACTIVATION_FUNCTION_GPU(activations[id]) + bias;
    }
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
        //       Signature: handel, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
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

        actiavtionFunction << <DimBlock, DimGrid >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, m_shape[layerNum]);
        
        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
    
    }


    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/