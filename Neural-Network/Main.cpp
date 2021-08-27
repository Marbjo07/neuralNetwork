#include "NeuralNetwork.h"


template<typename T>
void printVector(std::vector<T> vector) {
    for (int i : vector) {
        std::cout << vector[i] << " | ";
    }
    std::cout << std::endl;
}



int main() {

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Hello World" << std::endl;

    srand(0);


    NeuralNet model;
    float learningRate = 0.01;


    model.addLayer(3);
    model.addLayer(4);
    model.addLayer(5);
    model.addLayer(3);


    model.init();

    model.setInputs({ 0, 1, 2 });


    //for (auto i = 0; i < model.m_numberLayers; i++) {
    //    for (auto x = 0; x < model.m_layers[i].m_numberNeurons; x++) {
    //        printVector(model.m_layers[i].getWeights()[x]);
    //    }
    //    std::cout << "\n";
    //}


    printVector(model.feedForward());

    std::cout << "\n";


    for (auto x = 0; x < model.m_numberLayers; x++) {
        printVector(model.m_layers[x].getActivation());
    }


    std::cout << "Bye World" << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;




    return 1;

}