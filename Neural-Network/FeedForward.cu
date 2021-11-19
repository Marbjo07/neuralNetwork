#pragma once

#include "NeuralNetwork.hpp"
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\curand.h"
#include "E:\CUDA\Cuda Development\include\cublas_v2.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"
#include <chrono>


#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define BLOCK_SIZE 32


__global__ void gpu_matrix_mult(const float* a, const float* b, float* c, const int M, const int N, const int K) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    /*
    if ((Row < 1) && (Col < N)) {

        float Cvalue = 0;

        for (int k = 0; k < N; ++k) {

            Cvalue += a[Row * N + k] * b[k * M + Col];
        }
        c[Row * M + Col] = Cvalue;
    }*/
    if ((Row < 1) && (Col < N)) {
        float Cvalue = 0;

        for (int k = 0; k < M; k++) {
            Cvalue += a[k] * b[Col * M + k];
        }
        c[Col] = Cvalue;
    }
}

#define STEPSIZE 8
std::vector<float> NeuralNet::feedForward() {

    uint32_t neuronNum;
    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        std::vector<float> h_A = m_layers[layerNum - 1].getActivations();


        // reset values from previus feedforward
        // does it on every layer except inputlayer
        if (layerNum != 1) {
            for (size_t i = 0; i < m_layers[layerNum - 1].m_numberNeurons; i++) {
                //m_layers[layerNum - 1].m_neurons[i].m_activation = 0;
            }
        }
            
        std::vector<float> h_B;
        m_layers[layerNum].getWeights1D(&h_B);

        float* h_C = (float*)malloc(m_layers[layerNum].m_numberNeurons * sizeof(float));


        // Allocate 3 arrays on GPU
        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, h_A.size() * sizeof(float));
        cudaMalloc(&d_B, h_B.size() * sizeof(float));
        cudaMalloc(&d_C, m_layers[layerNum].m_numberNeurons * sizeof(float));

        cudaDeviceSynchronize();

        
        cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);


        dim3 DimGrid(512, 1, 1);
        dim3 DimBlock(512, 1, 1);
        gpu_matrix_mult << <DimGrid, DimBlock >> > (d_A, d_B, d_C, m_layers[layerNum-1].m_numberNeurons, m_layers[layerNum].m_numberNeurons, 1);

        cudaMemcpy(h_C, d_C, m_layers[layerNum].m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
            
        
        for (size_t i = 0; i < m_layers[layerNum].m_numberNeurons; i++) {
            m_layers[layerNum].m_neurons[i].m_activation = h_C[i];
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        neuronNum = 0;
        if (STEPSIZE < m_layers[layerNum].m_numberNeurons) {

            for (; neuronNum < m_layers[layerNum].m_numberNeurons - STEPSIZE; neuronNum += STEPSIZE) {

                m_layers[layerNum].m_neurons[neuronNum + 0].m_activation = m_layers[layerNum].m_neurons[neuronNum + 0].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 0].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 1].m_activation = m_layers[layerNum].m_neurons[neuronNum + 1].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 1].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 2].m_activation = m_layers[layerNum].m_neurons[neuronNum + 2].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 2].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 3].m_activation = m_layers[layerNum].m_neurons[neuronNum + 3].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 3].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 4].m_activation = m_layers[layerNum].m_neurons[neuronNum + 4].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 4].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 5].m_activation = m_layers[layerNum].m_neurons[neuronNum + 5].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 5].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 6].m_activation = m_layers[layerNum].m_neurons[neuronNum + 6].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 6].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 7].m_activation = m_layers[layerNum].m_neurons[neuronNum + 7].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 7].m_activation);

            }
        }
        for (; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {
            m_layers[layerNum].m_neurons[neuronNum].m_activation = m_layers[layerNum].m_neurons[neuronNum].activationFunction(m_layers[layerNum].m_neurons[neuronNum].m_activation);
        }

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