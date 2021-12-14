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

#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#define SIZEOF(x) sizeof(x) / sizeof(x[0])

#define ACTIVATION_FUNCTION(x) tanh(x)

#define GRID_SIZE_NEURALNETWORK 4
#define BLOCK_SIZE_NEURALNETWORK 8

#define CHECK_FOR_KERNEL_ERRORS(identifier) cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { std::cout << "Error in " << identifier << " "<< cudaGetErrorString(err) << std::endl; }

class NeuralNet {

public:
    class Layer {

    public:
        class ANN {

        public:

            uint32_t m_numberNeurons = 0;

            float d_bias = 1;
            float* d_weights;
            float* d_activations;

            // Every weight is set to defualtWeight if its not eqaul to NULL
            ANN(int numberOfNeurons, int numberOfNeuronsPrevLayer = 0, const float defualtWeight = NULL);

            void setActivation(std::vector<float> a);

            float* getActivations();

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
      
    void mutate(float mutationStrenght);

    // Simulates the neuralNet
    float* feedForward();

    // Dont call init after loading from a path
    void init(std::string name, const float defualtWeight = NULL);

    // Returns last layer activation
    float* getOutput();

    // Adds layer to neuralNet
    void addLayer(int numberOfNeurons);

    // Sets first layer to passed vector 
    // Prints error and returns if passed array length is'nt matching first layer size
    void setInput(std::vector<float> input);


    // Sets first layer to random values between -1 and 1
    void setRandomInput();
    
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

};

#include "Random.cuh"
#include "Tests.hpp"
#include "GpuHelperFunctions.cuh"

#endif // !NEURALNETWORK_HPP
