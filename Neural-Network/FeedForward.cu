#pragma once

#include "NeuralNetwork.cuh"


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

__global__ void activationFunction(float* activations, float bias, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


#pragma unroll
    for (; id < size; id += blockDim.x * blockDim.y) {

        float x = activations[id];

        switch (functionNum) {

        // sigmoid
        case 0:
            activations[id] = 1 / (1 + expf(-x));
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

__global__ void testFunc(float* ar, const int size) {
    for (int j = 0; j < size; j++) {
        printf("%.3f ", ar[j]);
    }
}

void NeuralNet::softMax() {
    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(GRID_SIZE_NEURALNETWORK, GRID_SIZE_NEURALNETWORK, 1);
    dim3 DimBlock(BLOCK_SIZE_NEURALNETWORK, BLOCK_SIZE_NEURALNETWORK, 1);
    softMax_gpu << <DimBlock, DimGrid, 0, m_deviceStream >> > (m_layers.back().d_activations, m_shape.back());
    CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
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

        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");


        int functionNum = getActivationFuncNum(layerNum);
        activationFunction << <DimGrid, DimBlock, 0, m_deviceStream >> > (m_layers[layerNum].d_activations, m_layers[layerNum].m_bias, functionNum, m_shape[layerNum]);
        CHECK_FOR_KERNEL_ERRORS("NEURALNET::FEEDFORWARD");
    }

 

    return this->getOutput();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/