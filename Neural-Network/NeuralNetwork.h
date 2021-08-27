#include <iostream>
#include <vector>
#include <random>
#include <chrono>

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
            float m_gradient = 0;

            Neuron(int numberOfNeuronsNextLayer);

        };

    public:

        int m_numberNeurons = 0;
        int m_numberNeuronsNextLayer = 0;

        std::vector<Neuron> m_neurons;

        Layer(bool createWeightsFlagg, int numberOfNeurons, int numberOfNeuronsNextLayer = 0);

        std::vector<std::vector<float>> getWeights();


        std::vector<float> getActivation();

        void setActivation(std::vector<float> a);
    };



public:

    std::vector<Layer> m_layers;
    std::vector<int> m_shape;
    int m_numberLayers;

    float derivativeOfSigmoid(float output);

    float sigmoid(float x);

    std::vector<float> dot(std::vector<float> x1, std::vector<std::vector<float>> x2);

    void addLayer(int numberOfNeurons);

    std::vector<float> feedForward();

    void setInputs(std::vector<float> inputs);

    void init();
};