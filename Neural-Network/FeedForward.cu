#pragma once

#include "NeuralNetwork.cuh"


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

// this is ugly, but its fast

__global__ void activationFunction_sigmoid(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#pragma unroll
    for (; id < size; id += gridDim.x * gridDim.y * blockDim.x * blockDim.y) {
        activations[id] = 1 / (1 + expf(-activations[id])) + bias;
    }
}

__global__ void activationFunction_relu(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#pragma unroll
    for (; id < size; id += gridDim.x * gridDim.y * blockDim.x * blockDim.y) {
        activations[id] = fmaxf(activations[id], 0) + bias;
    }
}

__global__ void activationFunction_tanh(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#pragma unroll
    for (; id < size; id += gridDim.x * gridDim.y * blockDim.x * blockDim.y) {
        activations[id] = tanh(activations[id]) + bias;
    }
}

__global__ void activationFunction_custom(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#pragma unroll
    for (; id < size; id += gridDim.x * gridDim.y * blockDim.x * blockDim.y) {
        activations[id] = 1 / (1 - activations[id] + activations[id] * activations[id]) + bias;
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
    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    softMax_gpu << <DimBlock, DimGrid, 0, m_deviceStream >> > (m_layers.back().d_activations, m_shape.back());
    CHECK_FOR_KERNEL_ERRORS;
}

float* NeuralNet::feedForward(uint32_t gridSize, uint32_t blockSize) {
    cudaSetDevice(m_deviceNum);


    if (gridSize == NULL || gridSize <= 0) {
        gridSize = m_gridFeedforward;
    }

    if (blockSize == NULL || blockSize <= 0) {
        blockSize = m_blockFeedforward;
    }

    dim3 DimGrid(gridSize, gridSize, 1);
    dim3 DimBlock(blockSize, blockSize, 1);

    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        GpuHelperFunc::cublasCompute(
            m_feedForwardHandle,
            m_layers[layerNum - 1].d_activations,
            m_layers[layerNum].d_weights,
            m_layers[layerNum].d_activations,
            m_shape[layerNum],
            1,
            m_shape[layerNum - 1],
            m_deviceNum
        );

        CHECK_FOR_KERNEL_ERRORS;



        int functionNum = getActivationFuncNum(layerNum);
        switch (functionNum) {

            // sigmoid
            case 0:
                activationFunction_sigmoid << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
                CHECK_FOR_KERNEL_ERRORS;
                break;


            // relu
            case 1:
                activationFunction_relu << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
                CHECK_FOR_KERNEL_ERRORS;
                break;

            // tanh
            case 2:
                activationFunction_tanh << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
                CHECK_FOR_KERNEL_ERRORS;
                break;
            
            // linear
            case 3:
                GpuHelperFunc::forEach::constVal::add << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, m_shape[layerNum]);
                break;

            // custom
            case 4:
                activationFunction_custom << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
                CHECK_FOR_KERNEL_ERRORS;
        }
    }

 

    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/