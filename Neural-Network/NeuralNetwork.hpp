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

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

class NeuralNet {

public:
    class Layer {

    public:

        class Neuron {
        public:

            // Connections from this neuron to the prev layers neurons
            std::vector<float> m_weights;
            float m_bias = 1;

            float m_activation = 0;

            float activationFunction(float x);

            //Every weight is set to defualtValue if spesified
            Neuron(std::mt19937* gen, const float* constVal, int numberOfNeuronsPrevLayer, int defualtValue);
        };


        int m_numberNeurons = 0;

        std::vector<Neuron> m_neurons;

        // Every weight is set to defualtWeight if not eqaul to NULL
        Layer(std::mt19937* gen, const float* constVal, int numberOfNeurons, int numberOfNeuronsPrevLayer = 0, int defualtWeight = NULL);

        void getWeights(std::vector<std::vector<float>>* weights);

        std::vector<float> getActivation();

        std::vector<float> getBias();

        void setActivation(std::vector<float>* a);

    };


    std::vector<Layer> m_layers;
    std::vector<int> m_shape;


    int m_numberLayers;
    int m_totalNumberOfNeurons;


    // Name of neuralNet
    // Is printed in warings;
    std::string m_name;

    // Every weight and bias is randomized
    void random();
      
    // Returns last layer output
    std::vector<float> feedForward();

    // Adds layer to neuralNet
    void addLayer(int numberOfNeurons);

    // Sets first layer to passed vector 
    // Prints error if passed vector length isnt matching first layer size
    // Resizes passed vector to length of first layer with defualt of 1
    void setInput(std::vector<float> input);


    // Initializes all values
    // Sets model name
    // Every weight is set to defualtWeight if its passed
    void init(std::string name, int zeroWeights = NULL);

    // 
    void naturalSelection(
        std::vector<float> target,
        int numberOfTest,
        float mutationStrength,
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
    void printWeightAndBias();


    // Mean absolute error
    float MAELossFunction(std::vector<float> output, std::vector<float> target);

    // Mean squard error
    float MSELossFunction(std::vector<float> output, std::vector<float> target);


    // Returns sum of weights and bias
    float sumOfWeightsAndBias();


};

#endif // !NEURALNETWORK_HPP
