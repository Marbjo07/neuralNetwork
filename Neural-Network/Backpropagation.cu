#pragma once

#include "NeuralNetwork.cuh"


#ifndef BACKPROPAGATION_CU
#define BACKPROPAGATION_CU


__global__ void computeDeltaHiddenLayers(float* delta, const float* nextDelta, const float* nextWeight, const float* activations, const int functionNum, const int size, const int nextSize) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (id == 0) {
        for (int j = 0; j < size; j++) {

            // compute error
            float error = 0;

            for (int neu = 0; neu < nextSize; neu++) {
                 error += nextWeight[j * nextSize + neu] * nextDelta[neu];
            }

            float x = activations[j];
            float deriv = 0;

            // sigmoid
            if (functionNum == 0) {
                deriv = x * (1.0 - x);
            }
            // relu
            else if (functionNum == 1) {
                deriv = x > 0.0 ? 1.0 : 0.0;
            }
            // tanh
            else if (functionNum == 2) {
                deriv = expf(2 * x) + 1.0;
                deriv = (4.0 / deriv) - (4.0 / (deriv * deriv));
            }
            // none
            else if (functionNum == 3) {
                deriv = 1.0;
            }
            // custom
            else if (functionNum == 4) {
                deriv = (2.0 / (expf(-x) + 1.0)) - 1.0;
            }
            
            delta[j] = error * deriv;
        }
    }
}

__global__ void computeDeltaLastLayer(float* delta, const float* activations, float* expected, const int functionNum, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {

        float x = activations[id];
        float tmp = 0;

        // sigmoid
        if (functionNum == 0) {
            tmp = x * (1 - x);
        }
        // relu
        else if (functionNum == 1) {
            tmp = x > 0 ? 1 : 0;
        }
        // tanh
        else if (functionNum == 2) {
            tmp = expf(2 * x) + 1;
            tmp = (4 / tmp) - (4 / (tmp * tmp));
        }
        // none
        else if (functionNum == 3) {
            tmp = 1;
        }
        // custom
        else if (functionNum == 4) {
            tmp = (2 / (expf(-x) + 1)) - 1;
        }

        //                |-------------------------| error
        delta[id] = tmp * (activations[id] - expected[id]);

    }
}

__global__ void weightUpdate(float* delta, float* prevLayerActivations, float* weights, float learningRate, const int prevSize, const int size) {

    int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (id == 0) {
        for (int neuronNum = 0; neuronNum < prevSize; neuronNum++) {
            for (int j = 0; j < size; j++) {
                 weights[neuronNum * size + j] -= learningRate * delta[j] * prevLayerActivations[neuronNum];
            }
        }
    }
}

void NeuralNet::backpropagation(const std::vector<std::vector<float>> dataset, const std::vector<std::vector<float>> correctOutput,
    const float updateWeightsAfterEveryBackPass,
    int batchSize,
    const bool randomBatching,
    const bool averageOutDeltas) {

    cudaSetDevice(m_deviceNum);

    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(1, 1, 1);

    float* d_expectedOutput = NULL;
    cudaMalloc(&d_expectedOutput, sizeof(float) * m_shape.back());

    int datasetIndex = 0;

    if (batchSize == NULL) {
        batchSize = dataset.size();
    }

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

        if (randomBatching) {
            datasetIndex = std::rand() % dataset.size();
        }
        else {
            datasetIndex = batchIndex;
        }

        setInput(dataset[datasetIndex]);

        feedForward();

        cudaMemcpy(d_expectedOutput, correctOutput[datasetIndex].data(), m_shape.back() * sizeof(float), cudaMemcpyHostToDevice);

        int funcNum = getActivationFuncNum(m_numberLayers - 1);
        computeDeltaLastLayer << <1, 1, 0, m_deviceStream >> > (m_layers.back().d_delta, m_layers.back().d_activations, d_expectedOutput, funcNum, m_layers.back().m_numberNeurons);
        CHECK_FOR_KERNEL_ERRORS("NEURALNET::backpropagation, computeDeltaLastLayer");

        for (int layerNum = m_numberLayers - 2; layerNum > 0; layerNum--) {

            int funcNum = getActivationFuncNum(layerNum);
            computeDeltaHiddenLayers << <1, 1, 0, m_deviceStream >> > (
                m_layers[layerNum].d_delta,
                m_layers[layerNum + 1].d_delta,
                m_layers[layerNum + 1].d_weights,
                m_layers[layerNum].d_activations,
                funcNum,
                m_shape[layerNum],
                m_shape[layerNum + 1]);

            CHECK_FOR_KERNEL_ERRORS("NEURALNET::backpropagation, computeDeltaHiddenLayers");
        }


        if (updateWeightsAfterEveryBackPass != NULL) {
            updateWeights(updateWeightsAfterEveryBackPass);
            clearDelta();
        }

    }
    if (averageOutDeltas) {
        for (int layerNum = m_numberLayers - 1; layerNum > 0; layerNum--) {
            GpuHelperFunc::forEach::constVal::div << <1, 1, 0, m_deviceStream >> > (m_layers[layerNum].d_delta, m_layers[layerNum].d_delta, (float)batchSize, m_shape[layerNum]);
            CHECK_FOR_KERNEL_ERRORS("NEURALNET::backpropagation, GpuHelperFunc::forEach::constVal::div");
        }
    }

    return;
}


void NeuralNet::updateWeights(float learning_rate) {
    
    cudaSetDevice(m_deviceNum);

    for (int layerNum = m_numberLayers - 1; layerNum > 0; layerNum--) {
        weightUpdate << <1, 1, 0, m_deviceStream >> > (
            m_layers[layerNum].d_delta,
            m_layers[layerNum - 1].d_activations,
            m_layers[layerNum].d_weights,
            learning_rate,
            m_shape[layerNum - 1],
            m_shape[layerNum]
            );
        CHECK_FOR_KERNEL_ERRORS("NEURALNET::updateWeights, weightUpdate");
    }

    return;
}

void NeuralNet::clearDelta() {
    cudaSetDevice(m_deviceNum);

    for (int layerNum = 0; layerNum < m_numberLayers; layerNum++) {
        GpuHelperFunc::setAllValuesInArrayToOneVal<<<1, BLOCK_SIZE_NEURALNETWORK, 0, m_deviceStream >>>(m_layers[layerNum].d_delta, m_shape[layerNum], 0);
        CHECK_FOR_KERNEL_ERRORS("NeuralNet::clearDelta()");
    }

    return;
}

#endif //!BACKPROPAGATION_CU