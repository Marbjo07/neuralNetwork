#pragma once

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

            float m_bias = 1;
            float* d_weights = NULL;
            float* d_activations = NULL;

            // Every weight is set to defualtWeight if its not eqaul to NULL
            ANN(uint64_t seed, int numberOfNeurons, int numberOfNeuronsPrevLayer = 0, const float defualtWeight = NULL);

            void setActivation(std::vector<float> a);

            float* getActivations();

        };
    };

    std::vector<Layer::ANN> m_layers;
    std::vector<uint32_t> m_shape;

    uint32_t m_numberLayers;
    uint32_t m_totalNumberOfNeurons;

    uint32_t m_gridFeedforward = 2;
    uint32_t m_blockFeedforward = 3;

    curandGenerator_t m_cudaRandGen;

    cublasHandle_t m_feedForwardHandle;

    // Name of neuralNet
    // Is printed in warings;
    std::string m_name;


    // Every weight and bias is randomized
    void random(uint64_t seed);
      
    void mutate(float mutationStrenght);

    // Simulates the neuralNet
    float* feedForward(uint32_t gridSize = NULL, uint32_t blockSize = NULL);

    // Dont call init after loading from a path
    void init(std::string name, int64_t seed, const float defualtWeight = NULL);

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


    // Print every activation
    // Not recomended on large models
    void printActivations();

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

    // Returns collective error of all the tests given
    float performTest(std::vector<std::vector<float>> testData, std::vector<std::vector<float>> expectedOutput);

    void optimizeParametersFeedforward(uint32_t maxGrid, uint32_t maxBlock, uint32_t numberOfTest);
    
};

#include "Random.cuh"
#include "Tests.hpp"
#include "GpuHelperFunctions.cuh"

#endif // !NEURALNETWORK_HPP
