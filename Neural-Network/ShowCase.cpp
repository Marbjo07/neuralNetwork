#include "NeuralNetwork.cuh"


int main() {
    
    // important to say hello
    printf("Hello World\n");


    auto t1 = std::chrono::high_resolution_clock::now();

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    std::vector< std::vector< float > > dataset = {};
    std::vector< std::vector< float > > labels= {};

    int numberOfDatapoints = 10;

    for (int i = 0; i < numberOfDatapoints; i++) {
        dataset.push_back({ 
            float(i) / numberOfDatapoints
        });

        labels.push_back({ 
            float(i) / numberOfDatapoints
        });
    }

    for (auto x : dataset) std::cout << x[0] << " ";
    printf("\n");
    for (auto x : labels) std::cout << x[0] << " ";
    printf("\n");

    float learning_rate = 0.1;


    NeuralNet model;


    model.m_shape = { 1, 100, 1 };
    model.m_activationFunctions = { "relu", "relu"};
    model.m_deviceNum = 0;

    // Makes all weights and bias
    // Init will clear model if already called!
    // If defualtWeight is specified every weight is set to that value
    // "AI" is the name of the model. The name is printed in warrnings
    
    model.init("AI", clock());


    printf("loss: %.6f\n", model.performTest(dataset, labels));

    model.printWeightsAndBias();


    for (int epoch = 0; epoch < 1000; epoch++) {

        if (epoch % 10 == 0) {
            printf("loss: %.6f\n", model.performTest(dataset, labels));
        }

#if 0
        model.backpropagation(dataset, labels, NULL, 0, 0, true);
        model.updateWeights(learning_rate);
        model.clearDelta();
#else
        model.backpropagation(dataset, labels, learning_rate);

#endif

    }

    printf("loss: %.6f\n", model.performTest(dataset, labels));

    for (int i = 0; i < dataset.size(); i++) {
        model.setInput(labels[i]);

        model.feedForward();

        printf("Input: %.3f, Correct Output: %.3f, Output: %.3f, Dif: %.3f\n", dataset[i][0], labels[i][0], model.getOutput()[0], abs(labels[i][0] - model.getOutput()[0]));
    }

    // Saving and Loading is not nesseary but its just shown here.
    
    // Saves the model to a bin file.
    // Loads a pretraind model
    // Just make a empty model and load from bin file
    // DO NOT call init after loading model because this will clear the loaded model.
    model.save(savePath);

    model.load(savePath);

    model.printWeightsAndBias();

    model.printActivations();

    model.printOutput();

    // Prints sum of weights and bias used in debuging.
    printf("Sum of weight and bias: %.6f\n", model.sumOfWeightsAndBias());

    printf("Duration in milliseconds: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());

    return 48879;
}