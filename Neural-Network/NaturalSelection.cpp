#include "NeuralNetwork.h"

template<typename T>
void printVector(std::vector<T> vector) {
    for (auto i : vector) {
        std::cout << i << " | ";
    }
}


float calcError(std::vector<float> output, std::vector<float> target) {
    float e{};
    for (auto i = 0; i < output.size(); i++)
        e += std::abs(target[i] - output[i]);
    return e / output.size();
}

NeuralNet combine(NeuralNet model1, NeuralNet model2) {

    for (auto layerNum = 0; layerNum < model1.m_numberLayers; layerNum++) {

        for (auto neuronNum = 0; neuronNum < model1.m_layers[layerNum].m_numberNeurons; neuronNum++) {
        
            for (auto weightNum = 0; weightNum < model1.m_layers[layerNum].m_neurons[neuronNum].m_weight.size(); weightNum++) {
            

                // Set weight to average of both models
                model1.m_layers[layerNum].m_neurons[neuronNum].m_weight[weightNum] += model2.m_layers[layerNum].m_neurons[neuronNum].m_weight[weightNum];
                model1.m_layers[layerNum].m_neurons[neuronNum].m_weight[weightNum] /= 2;
            }

            // Set bias to average of both models
            model1.m_layers[layerNum].m_neurons[neuronNum].m_bias += model2.m_layers[layerNum].m_neurons[neuronNum].m_bias;
            model1.m_layers[layerNum].m_neurons[neuronNum].m_bias /= 2;


        }

    }


    return model1;
}

void NeuralNet::naturalSelection(std::vector<float> target, int numberOfGenerations, int sizeOfGeneration) {

    NeuralNet bestModel = *this;
    NeuralNet tempModel;

    float lowestError = calcError(bestModel.feedForward(), target);

    for (auto gen = 0; gen < numberOfGenerations; gen++) {

        if (gen % 100 == 0) {
            std::cout << "Gen: " << gen << std::endl;
        }

        tempModel = bestModel;

        for (auto i = 0; i < sizeOfGeneration; i++) {


            for (auto i = 0; i < m_numberLayers; i++) {
                tempModel.m_layers[i].mutateThisLayer(1);
            }

            std::vector<float> output = tempModel.feedForward();
           
            float tempError = calcError(output, target);


            if (tempError < lowestError) {
                std::cout << "New best model: " << gen << " " << i << " with ";
                printVector(output);
                std::cout << "as output and " << tempError << " as error." << std::endl;

                tempModel.printWeightAndBias();

                lowestError = tempError;

                bestModel = combine(tempModel, bestModel);
                
                // start new generation
                break;
            }


        }
    }


    for (auto a = 0; a < m_numberLayers; a++) {
        m_layers[a].m_neurons = bestModel.m_layers[a].m_neurons;
    }

}
//0.351641 19491