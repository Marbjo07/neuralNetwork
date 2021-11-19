#include <iostream>

#include <filesystem> 


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


#include "../Neural-Network/NeuralNetwork.hpp"
#include "../Neural-Network/Tests.h"

class Image {
private:


public: 

	std::vector<float> data;
	int width;
	int height;


	void saveAsImageFile(std::string path, int width, int height) {
		for (uint32_t i = 0; i < data.size(); i++) {

			if (data[i] < 0) {
				data[i] = 0;
			}
			else {
				data[i] *= 255;
			}

			if (data[i] > 255)
			{
				data[i] = 255;
			}
		}

		uint8_t* rgb_image;
		rgb_image = (uint8_t*)malloc(width * height * 3);

		std::copy(data.begin(), data.end(), rgb_image);

		stbi_write_png(path.c_str(), width, height, 3, rgb_image, width * 3);

	}

};

Image loadVector(std::string path) {


	int width, height, depth;
	uint8_t* imageData = stbi_load(path.c_str(), &width, &height, &depth, 3);



	std::vector<float> output(&imageData[0], &imageData[width * height * depth]);

	Image outputImage;
	outputImage.data = output;
	outputImage.height = height;
	outputImage.width = width;


	std::cout << "Width: " << outputImage.width << "\t"
		<< "Height: " << outputImage.height << "\t"
		<< "Depth: " << depth << "\n";

	return outputImage;
}


#define SAVE_AND_LOAD_PATH_DISCRIMINATOR "E:/desktop/neuralNet/Discriminator0.bin"
#define SAVE_AND_LOAD_PATH_GENERATOR "E:/desktop/neuralNet/Generator0.bin"


int main() {

	Test::run(true, true);
	Test::FeedForwardBenchmark();
	return 0;

	std::cout << "Hello World\n";


	srand(time(NULL) * 12398239);
	int random = rand() % 109223859;
	std::cout << "seed: " << random << std::endl;
	srand(random);

	std::string path = "E:/Desktop/imageData/crop_part1";
	

	NeuralNet Generator;
	NeuralNet Discriminator;



	Discriminator.addLayer(200 * 200 * 3);
	Discriminator.addLayer(256);
	Discriminator.addLayer(64);
	Discriminator.addLayer(64);
	Discriminator.addLayer(16);
	Discriminator.addLayer(1);
	
	Discriminator.init("Discriminator", (const float)1);
	std::cout << "Discriminator init\n";
	
	Generator.addLayer(3);
	Generator.addLayer(16);
	Generator.addLayer(64);
	Generator.addLayer(64);
	Generator.addLayer(256);
	Generator.addLayer(200 * 200 * 3);
	
	
	Generator.init("Generator"); 
	
	std::cout << "Generator init\n";

	//Discriminator.load(SAVE_AND_LOAD_PATH_DISCRIMINATOR);
	//Generator.load(SAVE_AND_LOAD_PATH_GENERATOR);


	uint32_t discriminatorWin = 0;
	uint32_t generatorWin = 0;
	uint16_t numberOfTestInNaturalSelection = 100;
	Image generatorImage;
	generatorImage.height = 200;
	generatorImage.width = 200;
	std::vector<float> output;
	int i = 0;
	float prevSumOfWeightsAndBias;
	float sumOfWeightsAndBias = 0;

	float mutationStrengthGenerator = 100;
	float mutationStrengthDiscriminator = .1f;

	float quitThreshold = .1f;


	while (true) {
		for (const auto entry : std::filesystem::directory_iterator(path)) {
			std::cout << entry.path().generic_string() << std::endl;

			bool fakeFace = bool(rand() % 2);
			std::cout << "Fakeface: " << fakeFace << std::endl;


			Generator.setInput({0, 0, 0});

			if (fakeFace) {
				Discriminator.setInput(Generator.feedForward());

			}
			else {
				Discriminator.setInput(loadVector(entry.path().generic_string()).data);
			}


			output = Discriminator.feedForward();

			//for (auto x : output) std::cout << x << " | ";
			//std::cout << std::endl;

			// realface  | fakeface
			//   >0.5	 |   <0.5

			if (fakeFace) {

				if (output[0] >= 0.5) {
					std::cout << "Discrimitator thought it was real and it where not. :<\n";
					Discriminator.naturalSelection({ 0 }, numberOfTestInNaturalSelection, mutationStrengthDiscriminator, quitThreshold);

					generatorWin++;
				}
				else {
					std::cout << "Discrimitator thought it was fake and it was. :>\n";
					Generator.naturalSelection({ 1 }, numberOfTestInNaturalSelection, mutationStrengthGenerator, quitThreshold, &Discriminator);

					discriminatorWin++;
				}
			}
			else {

				if (output[0] >= 0.5) {
					std::cout << "Discrimitator thought it was real and it was. :>\n";
					Generator.naturalSelection({ 1 }, numberOfTestInNaturalSelection, mutationStrengthGenerator, quitThreshold, &Discriminator);

					discriminatorWin++;
				}
				else {
					std::cout << "Discrimitator thought it was fake but it where not. :<\n";
					Discriminator.naturalSelection({ 1 }, numberOfTestInNaturalSelection, mutationStrengthDiscriminator, quitThreshold);

					generatorWin++;
				}

			}

			std::cout << "Generator wins: " << generatorWin << " discrimitnator wins: " << discriminatorWin << std::endl;
			i++;

			if (i % 25 == 0) {
				Generator.setInput({ 0.5,0.5,0.5 });
				generatorImage.data = Generator.feedForward();
				generatorImage.saveAsImageFile("E:/desktop/aiOutput/image" + std::to_string(i) + ".png", generatorImage.width, generatorImage.height);
				Discriminator.save(SAVE_AND_LOAD_PATH_DISCRIMINATOR);
				Generator.save(SAVE_AND_LOAD_PATH_GENERATOR);

				prevSumOfWeightsAndBias = sumOfWeightsAndBias;
				sumOfWeightsAndBias = Generator.sumOfWeightsAndBias();

				std::cout << "Sum of weights and bias: " << sumOfWeightsAndBias << " previus sum of weights and bias: " << prevSumOfWeightsAndBias << std::endl;
			}

		}
	}
	return 0;

}



/*	for (auto i = 1; i < Discriminator.m_numberLayers; i++) {
		for (auto x = 0; x < Discriminator.m_layers[i].m_numberNeurons; x++) {
			std::cout << Discriminator.m_layers[i].m_neurons[x].m_activation << " ";
		}
		std::cout << "\n";
	}



			uint32_t neuronNum = 0;

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

		}*/