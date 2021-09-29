#pragma once

#include "NeuralNetwork.hpp"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP

__global__ void multiply(float* activations, float* weights, int* size, float* writeVal) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < *size) {
        (*writeVal) += activations[tid] + weights[tid];    
        tid += blockDim.x;      
    }
}


#define STEPSIZE 8
std::vector<float> NeuralNet::feedForward() {


    uint32_t neuronNum;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {
        float* prevLayerActivation;

        cudaMalloc(&prevLayerActivation, m_layers[layerNum - 1].m_numberNeurons * sizeof(float));

        cudaMemcpy(prevLayerActivation, m_layers[layerNum].getActivation().data(), m_layers[layerNum - 1].m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);

        for (neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {
            
            float* weights;


            cudaMalloc(&weights, m_layers[layerNum - 1].m_numberNeurons * sizeof(float));

            cudaMemcpy(weights, m_layers[layerNum].m_neurons[neuronNum].m_weights.data(), m_layers[layerNum - 1].m_numberNeurons * sizeof(float), cudaMemcpyHostToDevice);



            multiply<<<2,256>>>(prevLayerActivation, weights, &m_layers[layerNum - 1].m_numberNeurons, &m_layers[layerNum].m_neurons[neuronNum].m_activation);

            cudaDeviceSynchronize();
            cudaFree(weights);
        }

        cudaFree(prevLayerActivation);

        if (STEPSIZE < m_layers[layerNum].m_numberNeurons) {
            for (neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons - STEPSIZE; neuronNum += STEPSIZE) {

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

    return m_layers.back().getActivation();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP