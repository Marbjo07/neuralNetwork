#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

class NeuralNet {

private:

    class Layer {

    private:

        class Neuron {

        private:


        public:

            // Connections from this neuron to the next layers neurons
            std::vector<float> m_weight;
            float m_activation = 0;

            float m_bias = 1;

            //                                  if true weights is set to 0
            Neuron(int numberOfNeuronsNextLayer, bool zeroWeights = false);

            float activationFunction(float x);

            void mutateWeightAndBias(float mutationStrength);
        };

    public:

        int m_numberNeurons = 0;

        std::vector<Neuron> m_neurons;

        //                                                                                   if true weights is set to 0
        Layer(int numberOfNeurons, int numberOfNeuronsNextLayer = 0, bool zeroWeights = false);

        std::vector<std::vector<float>> getWeights();

        std::vector<float> getActivation();

        std::vector<float> getBias();
        
        void setActivation(std::vector<float> a);

        void mutateThisLayer(float mutationStrenght);

    };



public:

    std::vector<Layer> m_layers;
    std::vector<int> m_shape;

    int m_numberLayers;

    std::vector<float> dot(std::vector<float> x1, std::vector<std::vector<float>> x2);

    std::vector<float> feedForward();


    void addLayer(int numberOfNeurons);

    void setInputs(std::vector<float> inputs);

    void init(bool zeroWeights = false);

    void naturalSelection(std::vector<float> target, int numberOfGenerations, int sizeOfGeneration);
    
    void save(std::string path);

    void load(std::string path);

    void printWeightAndBias();
};