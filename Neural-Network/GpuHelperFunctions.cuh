#include "NeuralNetwork.cuh"


namespace GpuHelperFunc {

	__global__ void setAllValuesInArrayToOneVal(float* arrayToChange, const uint32_t size, const float val);

	__global__ void printArray(float* arrayToPrintfloat, const uint32_t size, const uint32_t newLine);

	__global__ void sumOfArray(float* arrayToSum, const uint32_t size, float sum);

};