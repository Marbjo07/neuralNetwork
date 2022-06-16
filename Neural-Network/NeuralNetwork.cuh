#pragma once

#include <math.h>
#include <chrono>
#include <fstream>
#include <future> 
#include <iostream>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

// cuda headers
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>


#include "MacrosAndDefinitions.h"

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP



class NeuralNet {

public:
    class Layer {

    public:

        class ANN {

        public:

            uint32_t m_numberNeurons = 0;

            float m_bias = 0;
            float* d_weights = NULL;
            float* d_activations = NULL;
            float* d_delta = NULL;
            float* d_newDelta = NULL;
            float* d_error = NULL;
            // Every weight is set to defualtWeight if its not eqaul to NULL
            ANN(const int deviceNum, int numberOfNeurons, int numberOfNeuronsPrevLayer = 0);

            void setActivation(const int deviceNum, std::vector<float> a);

            float* getActivations(const int deviceNum);

        };
    };

    std::vector<Layer::ANN> m_layers;

    std::vector<uint32_t> m_shape;
    std::vector<std::string> m_activationFunctions;

    uint32_t m_numberLayers;
    uint32_t m_totalNumberOfNeurons;

    uint32_t m_gridFeedforward = 2;
    uint32_t m_blockFeedforward = 3;

    curandGenerator_t m_cudaRandGen;

    cublasHandle_t m_feedForwardHandle;

    // handel running code on set gpu
    cudaStream_t m_deviceStream;

    // index of which gpu to run the model on
    int m_deviceNum = 0;

    // Name of neuralNet
    // Is printed in warings;
    std::string m_name;


    // Every weight and bias is randomized
    void random(uint64_t seed);
      
    void mutate(float mutationStrenght);

    // Simulates the neuralNet
    float* feedForward(uint32_t gridSize = NULL, uint32_t blockSize = NULL);

    void backpropagation(const std::vector<std::vector<float>> dataset, const std::vector<std::vector<float>> correctOutput,
        const float updateWeightsAfterEveryBackPass = (NULL),
        int batchSize = 0,
        const bool randomBatching = 0,
        const bool averageOutDeltas = false);

    void updateWeights(float learning_rate);

    void clearDelta();

    // Dont call init after loading from a path
    void init(std::string name, int64_t seed, const float weightDivisor = NULL, const float defualtWeight = NULL);
    
    // Returns last layer activation
    float* getOutput();

    // Adds layer to neuralNet
    void addLayer(int numberOfNeurons);

    // Sets first layer to passed vector 
    // Prints error and returns if passed array length is'nt matching first layer size
    void setInput(std::vector<float> input);


    // Sets first layer to random values between -1 and 1
    void setRandomInput(int64_t seed );
    
    // Saves model to binary file in location specified
    // Does not accept \ 
    void save(std::string path);

    // Loads model from binary file
    // Does not accept \ 
    void load(std::string path);


    // Prints every weight and bias
    // Not recomended on large models
    void printWeightsAndBias();


    // Prints every activation
    // Not recomended on large models
    void printActivations();

    // Prints delta for every neuron 
    void printDeltas();

    // Prints newdelta for every neuron 
    void printNewDeltas();

    // Prints output of model
    void printOutput();

    // Prints memory size of model
    void printSize();

    // Prints shape model
    void printShape();

    // Mean absolute error
    float MAELossFunction(float* output, std::vector<float> target);

    // Mean squard error
    float MSELossFunction(float* output, std::vector<float> target);

    // Overhead for all the lossfunctions
    float LossFunction(float* output, std::vector<float> target);

    // Returns sum of weights and bias
    float sumOfWeightsAndBias();

    float sumOfDeltas();

    // Returns collective error of all the tests given
    float performTest(std::vector<std::vector<float>> testData, std::vector<std::vector<float>> expectedOutput);

    void optimizeParametersFeedforward(uint32_t maxGrid, uint32_t maxBlock, uint32_t numberOfTest);
    
    // applies softmax on output layer
    void softMax();


    // returns the hash of activation function on that layer
    int getActivationFuncNum(const int layerNum);

};

#include "Random.cuh"
#include "Tests.hpp"
#include "GpuHelperFunctions.cuh"

#endif // !NEURALNETWORK_HPP
