#pragma once

#include "NeuralNetwork.hpp"
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\curand.h"
#include "E:\CUDA\Cuda Development\include\cublas_v2.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define BLOCK_SIZE 16
#define GRID_SIZE 1

__global__ void matrixMul(const float* activations, const float* weights, const float* bias, float* output, const int* shape, const int layerIndex) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * BLOCK_SIZE + col;

    while (id < shape[layerIndex]) {
        float tmp = 0;
        for (int i = 0; i < shape[layerIndex - 1]; i++) {
            tmp += activations[i] * weights[shape[layerIndex - 1] * id + i];
        }
        output[id] = ACTIVATION_FUNCTION_GPU(tmp) + bias[id];
        id += BLOCK_SIZE * BLOCK_SIZE;
    }
}


#define STEPSIZE 8
std::vector<float> NeuralNet::feedForward() {

    std::vector<float> h_A;
    int* d_Shape;
    cudaMalloc(&d_Shape, m_shape.size() * sizeof(int));


    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {


        h_A = m_layers[layerNum - 1].getActivations();


        std::vector<float> h_B;
        m_layers[layerNum].writeWeights1D(&h_B);

        float* h_C = (float*)malloc(m_layers[layerNum].m_numberNeurons * sizeof(float));



        // Allocate 3 arrays on GPU
        float* d_Activations, * d_Weights, * d_Bias, * d_Results;
        cudaMalloc(&d_Activations, m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_Weights, h_B.size() * sizeof(float));
        cudaMalloc(&d_Results, m_layers[layerNum].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_Bias, m_layers[layerNum].m_numberNeurons * sizeof(float));

        cudaMemcpyAsync(d_Activations, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Weights, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Shape, m_shape.data(), m_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Bias, m_layers[layerNum].getBias().data(), m_layers[layerNum].m_numberNeurons * sizeof(int), cudaMemcpyHostToDevice);

        dim3 DimGrid(GRID_SIZE, GRID_SIZE, 1);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

        matrixMul << <DimBlock, DimGrid >> > (d_Activations, d_Weights, d_Bias, d_Results, d_Shape, layerNum);

        cudaMemcpy(h_C, d_Results, m_layers[layerNum].m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);


        cudaDeviceSynchronize();

        for (size_t i = 0; i < m_layers[layerNum].m_numberNeurons; i++) {
            m_layers[layerNum].m_neurons[i].m_activation = h_C[i];
        }

        cudaFree(d_Activations);
        cudaFree(d_Weights);
        cudaFree(d_Bias);
    }

    if (m_name == "Generator") {
        for (uint32_t i = 0; i < m_layers.back().m_numberNeurons; i++) {

            m_layers.back().m_neurons[i].m_activation = std::min(m_layers.back().m_neurons[i].m_activation, 0.0f);

            m_layers.back().m_neurons[i].m_activation = std::max(m_layers.back().m_neurons[i].m_activation, 255.0f);
        }
    }

    return m_layers.back().getActivations();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP*/