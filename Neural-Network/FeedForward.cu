#pragma once

#include "NeuralNetwork.hpp"
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\curand.h"
#include "E:\CUDA\Cuda Development\include\cublas_v2.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define BLOCK_SIZE 64
#define GRID_SIZE 2

__global__ void matrixMul(const float* activations, const float* weights, const float* bias, float* output, const int* shape, const int layerIndex) {
    /*int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * BLOCK_SIZE + col;
    */
    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    while (id < shape[layerIndex]) {
        float tmp = 0;
        for (int i = 0; i < shape[layerIndex - 1]; i++) {
            tmp += activations[i] * weights[shape[layerIndex - 1] * id + i];
        }
        output[id] = ACTIVATION_FUNCTION_GPU(tmp) + bias[id];
        id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
}


float* NeuralNet::feedForward() {

    int* d_Shape;
    cudaMalloc(&d_Shape, m_shape.size() * sizeof(int));
    dim3 DimGrid(GRID_SIZE, GRID_SIZE, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        // Allocate 3 arrays on GPU
        float* d_Activations, * d_Weights, * d_Bias, * d_Results;
        cudaMalloc(&d_Weights,     m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_Activations,                                      m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_Results,     m_layers[layerNum].m_numberNeurons                                          * sizeof(float));
        cudaMalloc(&d_Bias,        m_layers[layerNum].m_numberNeurons                                          * sizeof(float));

        cudaMemcpyAsync(d_Weights,     m_layers[layerNum].m_weights,        m_layers[layerNum].m_numberNeurons * m_layers[layerNum - 1].m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Activations, m_layers[layerNum - 1].m_activation, m_layers[layerNum - 1].m_numberNeurons * sizeof(float),                                      cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Shape,       m_shape.data(),                      m_shape.size() * sizeof(int),                                                                cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Bias,        m_layers[layerNum].m_bias,           m_layers[layerNum].m_numberNeurons * sizeof(int),                                            cudaMemcpyHostToDevice);


        matrixMul << <DimBlock, DimGrid >> > (d_Activations, d_Weights, d_Bias, d_Results, d_Shape, layerNum);

        cudaDeviceSynchronize();

        cudaMemcpy(&m_layers[layerNum].m_activation[0], d_Results, m_layers[layerNum].m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

        //for (auto i = 0; i < m_layers[layerNum].m_numberNeurons; i++)  printf("%.6f ", m_layers[layerNum].m_activation[i]);

        cudaFree(d_Activations);
        cudaFree(d_Weights);
        cudaFree(d_Bias);
    }

    if (m_name == "Generator") {
        for (uint32_t i = 0; i < m_layers.back().m_numberNeurons; i++) {

            m_layers.back().m_activation[i] = std::min(m_layers.back().m_activation[i], 0.0f);

            m_layers.back().m_activation[i] = std::max(m_layers.back().m_activation[i], 255.0f);
        }
    }

    return m_layers.back().m_activation;
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/