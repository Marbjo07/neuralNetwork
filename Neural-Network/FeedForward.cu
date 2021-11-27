#pragma once

#include "NeuralNetwork.hpp"
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\curand.h"
#include "E:\CUDA\Cuda Development\include\cublas_v2.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

#define BLOCK_SIZE 10
//#define GRID_SIZE 1

__global__ void gpu_matrix_mult(const float* activations, const float* weights, float* output, const int* shape) {

    /*
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((Row < 1) && (Col < N)) {
        float Cvalue = 0;

        for (int k = 0; k < M; k++) {
            //(z* xMax* yMax) + (y * xMax) + x
            Cvalue += activations[k] * weights[(Col * M) + k];
        }
        output[Col] = Cvalue;
    }
    */
    /*
        //          block index
        int index = (blockIdx.y * blockDim.x + blockIdx.x) *

        //  number of threads pr block
            ((gridDim.x + 1) * gridDim.y) +

        //  thread index
            (threadIdx.y * gridDim.x + threadIdx.x);
    int numberOfAvailableThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    */

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    /*printf("Blk: (%d,%d) Thread: (%d,%d) => Id = %d  (%d, %d) \n",
        blockIdx.x, blockIdx.y,
        threadIdx.x, threadIdx.y,
        index,
        gridDim.x, gridDim.y);
    */

    __shared__ float sharedMemory[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int colCounter = Col; colCounter < shape[1]; colCounter += BLOCK_SIZE) {

        if (Col < shape[1] && Row < shape[0]) {
            float tmp = 0;
            for (int k = Row; k < shape[0]; k += blockDim.x) {
                //(z* xMax* yMax) + (y * xMax) + x
                tmp += activations[k] * weights[(colCounter * shape[0]) + k];
            }
            sharedMemory[Col][Row] = tmp;

        }

        __syncthreads();

        if (Row == 0 && Col < shape[1]) {
            float tmp = 0;
            for (int i = 0; i < shape[0]; i++) {
                tmp += sharedMemory[Col][i];
            }
            output[colCounter] = ACTIVATION_FUNCTION_GPU(tmp);

        }
        __syncthreads();
    }

}



#define STEPSIZE 8
std::vector<float> NeuralNet::feedForward() {

    std::vector<float> h_A;
    
    for (size_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {


        h_A = m_layers[layerNum - 1].getActivations();


        std::vector<float> h_B;
        m_layers[layerNum].writeWeights1D(&h_B);

        float* h_C = (float*)malloc(m_layers[layerNum].m_numberNeurons * sizeof(float));



        // Allocate 3 arrays on GPU
        float* d_A, * d_B, * d_C;
        int* d_Shape;
        cudaMalloc(&d_A, m_layers[layerNum - 1].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_B, h_B.size() * sizeof(float));
        cudaMalloc(&d_C, m_layers[layerNum].m_numberNeurons * sizeof(float));
        cudaMalloc(&d_Shape, m_shape.size() * sizeof(int));

        cudaDeviceSynchronize();


        cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Shape, m_shape.data(), m_shape.size() * sizeof(int), cudaMemcpyHostToDevice);

        //dim3 DimGrid(GRID_SIZE, GRID_SIZE, 1);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
        gpu_matrix_mult << <1, DimBlock >> > (d_A, d_B, d_C, d_Shape);

        cudaMemcpy(h_C, d_C, m_layers[layerNum].m_numberNeurons * sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();


        for (size_t i = 0; i < m_layers[layerNum].m_numberNeurons; i++) {
            m_layers[layerNum].m_neurons[i].m_activation = h_C[i];
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
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