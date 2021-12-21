#pragma once

#include "NeuralNetwork.cuh"


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define MAX (gridDim.x * gridDim.y * blockDim.x * blockDim.y)

__global__ void matrixMul(const float* activations, const float* weights, const float bias, float* output, const int currentSize, const int prevSize) {
    
    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int size = floor((float)currentSize / MAX);


    // Distrubute the remaining activations to the last few threads
    if (id == MAX - 1) {
        size += currentSize % MAX;
    }

    id *= size;
    int end = id + size;

    for (; id < end; ++id) {
        float tmp = 0;
        int i = 0;
        for (; i < prevSize - 4; i += 4) {
            float4 a_tmp = reinterpret_cast<const float4*>(&activations[i])[0];

            tmp += a_tmp.x * weights[prevSize * id + i];
            tmp += a_tmp.y * weights[(prevSize + 1) * id + i];
            tmp += a_tmp.z * weights[(prevSize + 2) * id + i];
            tmp += a_tmp.w * weights[(prevSize + 3) * id + i];

        }
        for (; i < prevSize; i++) {
            tmp += activations[i] * weights[prevSize * id + i];
        }
        output[id] = ACTIVATION_FUNCTION_GPU(tmp) + bias;
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



    for (size_t layerNum = 1; layerNum < m_numberLayers; ++layerNum) {
       
        matrixMul << <DimBlock, DimGrid >> > (
            m_layers[layerNum - 1].d_activations, 
            m_layers[layerNum].d_weights, 
            m_layers[layerNum].m_bias, 
            m_layers[layerNum].d_activations, 
            m_layers[layerNum].m_numberNeurons, 
            m_layers[layerNum - 1].m_numberNeurons);

        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
    
        cudaDeviceSynchronize();
        
    }
    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/