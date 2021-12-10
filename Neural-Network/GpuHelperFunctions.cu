
#include "GpuHelperFunctions.cuh"

namespace GpuHelperFunc {

	__global__ void setAllValuesInArrayToOneVal(float* arrayToChange, const uint32_t size, const float val) {

		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		while (id < size) {
			arrayToChange[id] = val;
			id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
		}

		return;
	}

	__global__ void printArray(float* arrayToPrint, const uint32_t size, const uint32_t newLine) {

		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		if (id == 0) {
			for (uint32_t i = 0; i < size; i++) {
				printf("%.6f ", arrayToPrint[i]);
				if (i % newLine == 0) {
					printf("\n");
				}
			}
			printf("\n");
		}
		else if (id == 1) {
			printf("No need to use more than one thread in GpuHelperFunc::printArray()\n");
		}
	}

	__global__ void sumOfArray(float* arrayToSum, const uint32_t size, float sum) {

		uint32_t id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


		if (id == 0) {
			float tmp = 0;
			for (uint32_t i = 0; i < size; i++) {
				tmp += arrayToSum[i];
			}
			sum = tmp;
		}
		else if (id == 1) {
			printf("No need to use more than one thread in GpuHelperFunc::sumOfArray()\n");
		}




	}

};