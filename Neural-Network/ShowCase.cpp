#include "NeuralNetwork.cuh"


int main() {

    auto t1 = std::chrono::high_resolution_clock::now();

    srand((uint32_t)time(NULL));

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    NeuralNet model;



    // Adds a virtuell layer with N number of neurons.
    // 
    // models shape:
    // o 
    //   \
    // o - o - o
    //   /
    //  o 

    // o is a neuron
    // \, / or - is a connection


    model.m_shape = { 3, 1 , 3 };
    

    // Check if feedforward is working correctly
    Test::run(true, false);


    // Makes all weights and bias
    // Init will clear model if allredy called!!
    // If defualtWeight is specified every weight is set to that value
    // "AI" is the name of the model. The name is printed in warrnings
    model.init("AI");


    // Check time for feedforward
    Test::FeedForwardBenchmark(model.m_shape);

    // Prints and uses the best Grids and Blocks value for feedforward
    model.optimizeGridsAndBlocksFeedforward(5, 32, 10);
    
    // Check time for new blocks and grid.
    // To keep these changes change m_gridFeedforward, m_blockFeedforward to the printed values.
    // The changes aren't big for small models but for bigger model the speed increase can be 
    Test::FeedForwardBenchmark(model.m_shape);

    model.printWeightsAndBias();
    

    // Run model
    model.feedForward();


    model.getOutput();


    std::vector< std::vector< float > > inputs        = { {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10} };
    std::vector< std::vector< float > > correctOutput = { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0} };
    

    NeuralNet tmpModel;
    tmpModel = model;

    float lowestError = model.performTest(inputs, correctOutput);
    float error = lowestError;
    for (auto j = 0; j < (2 << 9); j++) {

        tmpModel.random();

        error = tmpModel.performTest(inputs, correctOutput);
        if (error < lowestError) {
            lowestError = error;
            model = tmpModel;
            printf("error: %.6f\n", lowestError);
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

    std::vector<float> input = { 1 };

    model.setInput(input);

    model.feedForward();

    model.printOutput();


    // Prints sum of weights and bias used in debuging.
    printf("Sum of weight and bias: %.6f\n", model.sumOfWeightsAndBias());

    printf("Duration in milliseconds: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());

    return 1;
}