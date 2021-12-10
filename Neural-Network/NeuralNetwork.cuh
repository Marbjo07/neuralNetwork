#pragma once


#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>
#include <thread>
#include <future>
#include <stdlib.h>

#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\curand.h"
#include "E:\CUDA\Cuda Development\include\cublas_v2.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#define GRID_SIZE_NEURALNETWORK 2
#define BLOCK_SIZE_NEURALNETWORK 64

class NeuralNet {

public:
    class Layer {

    public:
        class ANN {

        public:

            uint32_t m_numberNeurons = 0;

            float d_bias = 0;
            float* d_weights;
            float* d_activations;

            // Every weight is set to defualtWeight if its not eqaul to NULL
            ANN(int numberOfNeurons, int numberOfNeuronsPrevLayer = 0, const float defualtWeight = NULL);

            void setActivation(float* a);
            

            float activationFunction(float x);
        };
    };


    std::vector<Layer::ANN> m_layers;
    std::vector<uint32_t> m_shape;

    uint32_t m_numberLayers;
    uint32_t m_totalNumberOfNeurons;


    // Name of neuralNet
    // Is printed in warings;
    std::string m_name;


    // Every weight and bias is randomized
    void random();
      

    // Simulates the neuralNet
    float* feedForward();


    // Returns last layer activation
    float* getOutput();

    // Adds layer to neuralNet
    void addLayer(int numberOfNeurons);

    // Sets first layer to passed vector 
    // Prints error and returns if passed array length is'nt matching first layer size
    void setInput(float* input, const size_t size);


    // Sets first layer to random values between -1 and 1
    void setRandomInput();

    // Initializes all values
    // Sets model name
    // Every weight is set to defualtWeight if its passed
    void init(std::string name, const float defualtWeight = NULL);

    // 
    void naturalSelection(
        std::vector<float> target,
        int numberOfTest,
        float mutationStrength,
        float quitThreshold,
        NeuralNet* checkerModel = NULL
    );
    
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

    // Mean absolute error
    float MAELossFunction(float* output, std::vector<float> target);

    // Mean squard error
    float MSELossFunction(float* output, std::vector<float> target);


    // Returns sum of weights and bias
    float sumOfWeightsAndBias();

};

#include "Macros.hpp"
#include "Random.cuh"
#include "Tests.hpp"
#include "GpuHelperFunctions.cuh"

#endif // !NEURALNETWORK_HPP
