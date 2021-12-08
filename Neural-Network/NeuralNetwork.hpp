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


#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

class NeuralNet {

public:
    class Layer {

    public:
        class ANN {

        public:

            uint32_t m_numberNeurons = 0;

            float* m_bias;
            float* m_weights;
            float* m_activation;

            // Every weight is set to defualtWeight if its not eqaul to NULL
            ANN(int numberOfNeurons, int numberOfNeuronsPrevLayer = 0, const float defualtWeight = NULL);

            void writeWeights(std::vector<std::vector<float>>* weights);

            void setActivation(float* a);

            float activationFunction(float x);
        };
    };


    std::vector<Layer::ANN> m_layers;
    std::vector<int> m_shape;


    uint32_t m_numberLayers;
    uint32_t m_totalNumberOfNeurons;


    // Name of neuralNet
    // Is printed in warings;
    std::string m_name;


    // Every weight and bias is randomized
    void random();
      
    // Returns last layer output after simulating the neuralNet
    float* feedForward();

    // Adds layer to neuralNet
    void addLayer(int numberOfNeurons);

    // Sets first layer to passed vector 
    // Prints error and returns if passed array length is'nt matching first layer size
    void setInput(float* input);


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
#include "Random.hpp"
#include "Tests.hpp"


#endif // !NEURALNETWORK_HPP
