#pragma once

#include "NeuralNetwork.cuh"


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define GRID_SIZE_FEEDFORWARD 4
#define BLOCK_SIZE_FEEDFORWARD 8

#define MAX gridDim.x * gridDim.y * blockDim.x * blockDim.y

__global__ void matrixMul(const float* activations, const float* weights, const float bias, float* output, const int currentSize, const int prevSize) {
    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    
    while (id < currentSize) {
        float tmp = 0;
        for (int i = 0; i < prevSize; i++) {
            tmp += activations[i] * weights[prevSize * id + i];
        }
        output[id] = ACTIVATION_FUNCTION(tmp) + bias;
        id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }

}


float* NeuralNet::feedForward() {

    dim3 DimGrid(GRID_SIZE_FEEDFORWARD, GRID_SIZE_FEEDFORWARD, 1);
    dim3 DimBlock(BLOCK_SIZE_FEEDFORWARD, BLOCK_SIZE_FEEDFORWARD, 1);

    /*float sum = 0;
    for (size_t i = 1; i < m_shape.size(); i++) {
        sum += m_shape[i - 1] * m_shape[i];
    }
    printf("%.6f\n", sum / (GRID_SIZE_FEEDFORWARD * GRID_SIZE_FEEDFORWARD * BLOCK_SIZE_FEEDFORWARD * BLOCK_SIZE_FEEDFORWARD));*/
    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
       
        matrixMul << <DimBlock, DimGrid >> > (
            m_layers[layerNum - 1].d_activations, 
            m_layers[layerNum].d_weights, 
            m_layers[layerNum].d_bias, 
            m_layers[layerNum].d_activations, 
            m_layers[layerNum].m_numberNeurons, 
            m_layers[layerNum - 1].m_numberNeurons);

        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
    
        cudaDeviceSynchronize();
    }

    if (m_name == "Generator") {
        for (uint32_t i = 0; i < m_layers.back().m_numberNeurons; i++) {

            m_layers.back().d_activations[i] = std::min(m_layers.back().d_activations[i], 0.0f);

            m_layers.back().d_activations[i] = std::max(m_layers.back().d_activations[i], 255.0f);
        }
    }
    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/