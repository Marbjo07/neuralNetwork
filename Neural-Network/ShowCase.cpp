#include "NeuralNetwork.cuh"

void showCase() {

    auto t1 = std::chrono::high_resolution_clock::now();

    srand((uint32_t)time(NULL));

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    int numberOfMutations = 50000;
    float mutationStrength = 1;

    NeuralNet model;

    std::cout << "Loaded model\n";


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


    model.m_shape = { 1, 3, 1 };


    // Makes all weights and bias
    // Init will clear model if allredy called!!
    // If defualtWeight if specified every weight is set to that value
    // "AI" is the name of the model. The name is printed in warrnings
    model.init("AI");

    // Output of model
    float* output = model.feedForward();

    // Print output
    std::cout << "Output: ";
    for (auto i = 0; i < SIZEOF(output);i++) std::cout << output[i] << " | ";
    std::cout << "\n";


    for (auto j = 0; j < 5; j++) {
        for (float i = 0; i < 10; i++) {

            // Sets input
            std::vector<float> x = { i };
            model.setInput(x.data(), x.size());


            // 1. Mutates the original model.
            // 2. If the mutation is better than the original the mutation is now the original.
            // 3. Error is calculated by MSE or if checkerModel is specified do step 4
            // 4. Error is calculated by MSE(target and output of checkerModel with input of output of this model)
            // 5. Do step 1 through 4 numberOfMutations times.
            model.naturalSelection({ i }, numberOfMutations, mutationStrength, 0);


            std::cout << "Target: " << i << " Output: " << model.feedForward()[0] << std::endl;
        }
    }


    // Saving and Loading is not nesseary but its just shown here.
    
    // Saves the model to a bin file.
    model.save(savePath);

    // Loads a pretraind model
    // Just make a empty model and load from bin file
    // DO NOT call init after loading model because this will clear the loaded model.
    model.load(savePath);

    output = model.feedForward();

    std::cout << "Output: ";
    for (auto i = 0; i < SIZEOF(output); i++) std::cout << output[i] << " | ";
    std::cout << "\n";

    // Prints sum of weights and bias used in debuging.
    std::cout << model.sumOfWeightsAndBias() << std::endl;


    std::cout << "Duration in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;
}