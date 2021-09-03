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

            //                                  if true weights is set to 0
            Neuron(int numberOfNeuronsNextLayer, bool zeroWeights = false);

            float sigmoid(float x);

            void mutateWeight(float mutationStrength);

        };

    public:

        int m_numberNeurons = 0;

        std::vector<Neuron> m_neurons;


        //                                                                                   if true weights is set to 0
        Layer(bool createWeightsFlagg, int numberOfNeurons, int numberOfNeuronsNextLayer = 0, bool zeroWeights = false);

        std::vector<std::vector<float>> getWeights();


        std::vector<float> getActivation();

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

    void naturalSelection(NeuralNet *pointerToOrig, std::vector<float> target, int numberOfGenerations, int sizeOfGeneration, float mutationStrenght);
    
    void save(std::string path);

    void load(std::string path);
};