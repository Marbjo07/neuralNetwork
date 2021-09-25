#pragma

#include "NeuralNetwork.hpp"

#ifndef FEEDFORWARD_CPP
#define FEEDFORWARD_CPP


#define STEPSIZE 8


std::vector<float> NeuralNet::feedForward() {


    uint32_t weightNum;
    uint32_t neuronNum;

    for (uint32_t layerNum = 1; layerNum < m_numberLayers; layerNum++) {

        for (uint32_t neuronNum = 0; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {

            weightNum = 0;
            if (STEPSIZE < m_layers[layerNum].m_neurons[neuronNum].m_weights.size()) {
                for (; weightNum < m_layers[layerNum].m_neurons[neuronNum].m_weights.size() - STEPSIZE; weightNum += STEPSIZE) {

                    m_layers[layerNum].m_neurons[neuronNum].m_activation += (


                        m_layers[layerNum - 1].m_neurons[weightNum + 0].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 0] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 1].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 1] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 2].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 2] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 3].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 3] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 4].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 4] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 5].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 5] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 6].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 6] +
                        m_layers[layerNum - 1].m_neurons[weightNum + 7].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum + 7]

                        );
                }
            }
            for (; weightNum < m_layers[layerNum].m_neurons[neuronNum].m_weights.size(); weightNum++) {
                m_layers[layerNum].m_neurons[neuronNum].m_activation +=
                    m_layers[layerNum - 1].m_neurons[weightNum].m_activation * m_layers[layerNum].m_neurons[neuronNum].m_weights[weightNum];
            }
        }

        neuronNum = 0;
        if (STEPSIZE < m_layers[layerNum].m_numberNeurons) {
            for (; neuronNum < m_layers[layerNum].m_numberNeurons - STEPSIZE; neuronNum += STEPSIZE) {

                m_layers[layerNum].m_neurons[neuronNum + 0].m_activation = m_layers[layerNum].m_neurons[neuronNum + 0].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 0].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 1].m_activation = m_layers[layerNum].m_neurons[neuronNum + 1].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 1].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 2].m_activation = m_layers[layerNum].m_neurons[neuronNum + 2].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 2].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 3].m_activation = m_layers[layerNum].m_neurons[neuronNum + 3].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 3].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 4].m_activation = m_layers[layerNum].m_neurons[neuronNum + 4].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 4].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 5].m_activation = m_layers[layerNum].m_neurons[neuronNum + 5].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 5].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 6].m_activation = m_layers[layerNum].m_neurons[neuronNum + 6].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 6].m_activation);
                m_layers[layerNum].m_neurons[neuronNum + 7].m_activation = m_layers[layerNum].m_neurons[neuronNum + 7].activationFunction(m_layers[layerNum].m_neurons[neuronNum + 7].m_activation);
            }
        }
        for (; neuronNum < m_layers[layerNum].m_numberNeurons; neuronNum++) {

            m_layers[layerNum].m_neurons[neuronNum].m_activation = m_layers[layerNum].m_neurons[neuronNum].activationFunction(m_layers[layerNum].m_neurons[neuronNum].m_activation);

        }
    }


    return m_layers.back().getActivation();
}
#undef STEPSIZE

#endif // !FEEDFORWARD_CPP
