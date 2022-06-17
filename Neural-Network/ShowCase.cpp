#include "NeuralNetwork.cuh"


int main() {

    Test::runBenchmarks();

    return 1;

    // important to say hello
    printf("Hello World\n");


    auto t1 = std::chrono::high_resolution_clock::now();

    std::string savePath = "E:/desktop/neuralNet/a.bin";

    std::vector< std::vector< float > > dataset = {};
    std::vector< std::vector< float > > labels= {};

    const int numberOfDatapoints = 10;

    const int numberOfInputsAndOutputs = 1;

    const int batchSize = numberOfDatapoints / 3;

    const int trainingMethod = 0; // SGD, GD, RMBGD

    float learning_rate = 0.1;

    
    for (int i = 0; i < numberOfDatapoints; i++) {
        
        dataset.push_back({});

        for (int j = 0; j < numberOfInputsAndOutputs; j++) {
            dataset[i].push_back(float(i + 1) / numberOfDatapoints);
        }

        labels.push_back({});

        for (int j = 0; j < numberOfInputsAndOutputs; j++) {
            labels[i].push_back(std::sin(dataset[i][j]));
        }
    }

    for (int i = 0; i < numberOfDatapoints; i++) {
        printf("Input:\t\t{");
       
        for (auto& x : dataset[i]) {
            printf("%.3f ", x);
        }
        printf("}\nCorrect Output:\t{");

        for (auto& x : labels[i]) {
            printf("%.3f ", x);
        }
        printf("}\n");

    }

    NeuralNet model;


    model.m_shape = { numberOfInputsAndOutputs, 256, numberOfInputsAndOutputs };
    model.m_activationFunctions = { "sigmoid", "sigmoid"};
    model.m_deviceNum = 1;

    // Makes all weights and bias
    // Init will clear model if already called!
    // If defualtWeight is specified every weight is set to that value 
    // "AI" is the name of the model. The name is printed in warrnings
    model.init("AI", 1, 5);
    //0.017424156889


    printf("loss: %.6f\n", model.performTest(dataset, labels));

    float prevLoss = 0;
    float loss = 0;

    for (int epoch = 0; epoch < 5000; epoch++) {

        if (epoch % 100 == 0) {
            prevLoss = loss;
            loss = model.performTest(dataset, labels);
            printf("epoch: %d \tloss: %.12f \tdif: %.12f\n",epoch, loss, loss - prevLoss);
        }

        if (trainingMethod == 0) {
            // Stochastic Gradient Descent
            model.backpropagation(dataset, labels, learning_rate);
            
        }
        else if (trainingMethod == 1) {
            // Gradient Descent 
            model.backpropagation(dataset, labels, NULL, 0, false, true);

        }
        else if (trainingMethod == 2) {
            // Random Mini Batch Gradient Descent
            model.backpropagation(dataset, labels, NULL, batchSize, true);

        }

        if (trainingMethod != 0) {
            // update weights
            // !Dont use this if you use Stochastic Gradient Descent!
            model.updateWeights(learning_rate);

            // clear delta after updating weights
            // !Dont use this if you use Stochastic Gradient Descent!
            model.clearDelta();

        }
    }
    return 0;
    printf("loss: %.6f\n", model.performTest(dataset, labels));

    for (int i = 0; i < dataset.size(); i++) {
        model.setInput(dataset[i]);

        model.feedForward();


        printf("Input:\t\t{");

        for (auto& x : dataset[i]) {
            printf("%.3f ", x);
        }
        printf("}\nCorrect Output:\t{");

        for (auto& x : labels[i]) {
            printf("%.3f ", x);
        }
        printf("}\nOutput:\t\t{");
        
        GpuHelperFunc::usePrintArrayFromCppFile(model.m_layers.back().d_activations, model.m_shape.back(), model.m_deviceNum, model.m_deviceStream);

        printf("}\nDif:\t\t{");

        for (int j = 0; j < model.m_shape.back(); j++) {
        
            printf("%.3f ", abs(labels[i][j] - model.getOutput()[j]) );
            
        }
        printf("}\n-----------\n");
    }

    // Saving and Loading is not nesseary but its just shown here.
    
    // Saves the model to a bin file.
    // Loads a pretraind model
    // Just make a empty model and load from bin file
    // DO NOT call init after loading model because this will clear the loaded model.
    model.printWeightsAndBias();

    model.save(savePath);

    model.load(savePath);

    model.printWeightsAndBias();

    printf("Duration in milliseconds: %lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count());

    return 48879;
}