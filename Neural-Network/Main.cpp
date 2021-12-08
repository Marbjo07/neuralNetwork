#include <iostream>

#include <filesystem> 


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


#include "../Neural-Network/NeuralNetwork.hpp"

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

	Test::run(true, false);
	Test::FeedForwardBenchmark();
	Test::InitBenchmark();
	return 0;

	std::cout << "Hello World\n";


	srand(time((time_t*)(NULL)) * 12398239);
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
	uint16_t numberOfTestInNaturalSelection = 10;
	Image generatorImage;
	generatorImage.height = 200;
	generatorImage.width = 200;
	float* output;
	int i = 0;
	float prevSumOfWeightsAndBias;
	float sumOfWeightsAndBias = 0;

	float mutationStrengthGenerator = 100;
	float mutationStrengthDiscriminator = .1f;

	float quitThreshold = .1f;
	std::vector<float> sampelInput = { 0.5, 0.5,0.5 };

	while (true) {
		for (const auto entry : std::filesystem::directory_iterator(path)) {
			std::cout << entry.path().generic_string() << std::endl;

			bool fakeFace = bool(rand() % 2);
			std::cout << "Fakeface: " << fakeFace << std::endl;


			Generator.setInput(sampelInput.data());

			if (fakeFace) {
				Discriminator.setInput(Generator.feedForward());

			}
			else {
				Discriminator.setInput(loadVector(entry.path().generic_string()).data.data());
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
				
				Generator.setInput(sampelInput.data());
				output = Generator.feedForward();

				generatorImage.data.clear();
				generatorImage.data.insert(generatorImage.data.end(), &output[0], &output[SIZEOF(output)]);

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