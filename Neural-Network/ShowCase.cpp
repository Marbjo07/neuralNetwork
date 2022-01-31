#include "NeuralNetwork.cuh"


int main() {
    
    // important to say hello
    printf("Hello World\n");


    auto t1 = std::chrono::high_resolution_clock::now();

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    NeuralNet model;


    model.m_shape = { 1, 2, 3, 2, 2 };
    model.m_activationFunctions = { "relu", "sigmoid", "sigmoid", "tanh" };
    
    // Makes all weights and bias
    // Init will clear model if already called!
    // If defualtWeight is specified every weight is set to that value
    // "AI" is the name of the model. The name is printed in warrnings
    
    model.init("AI", clock());

    model.feedForward();

    model.printOutput();

    model.softMax();

    model.printOutput();

    Test::runTests(true, false);
    
    Test::runBenchmarks();

    std::vector< std::vector< float > > inputs        = { {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10} };
    std::vector< std::vector< float > > correctOutput = { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0} };
    

    NeuralNet tmpModel;
    tmpModel = model;

    float lowestError = model.performTest(inputs, correctOutput);
    float error = lowestError;
    for (auto j = 0; j < (2 << 5); j++) {

        tmpModel.random(std::rand());

        error = tmpModel.performTest(inputs, correctOutput);
        printf("error: %.6f\n", error);

        if (error < lowestError) {
            lowestError = error;
            model = tmpModel;
        }

    }


    // Saving and Loading is not nesseary but its just shown here.
    
    // Saves the model to a bin file.
    // Loads a pretraind model
    // Just make a empty model and load from bin file
    // DO NOT call init after loading model because this will clear the loaded model.
    model.save(savePath);

    model.load(savePath);

    //model.printWeightsAndBias();

    model.setRandomInput(1);

    model.feedForward();


    model.printActivations();
    model.printOutput();

    // Prints sum of weights and bias used in debuging.
    printf("Sum of weight and bias: %.6f\n", model.sumOfWeightsAndBias());

    printf("Duration in milliseconds: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());

    return 48879;
}