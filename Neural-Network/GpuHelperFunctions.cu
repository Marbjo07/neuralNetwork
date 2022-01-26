
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

	void cublasCompute(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int uiWB, int uiHA, int uiWA) {
		float alpha = 1;
		float beta = 0;

		cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			uiWB,
			uiHA,
			uiWA,
			&alpha,
			d_B,
			uiWB,
			d_A,
			uiWA,
			&beta,
			d_C,
			uiWB
		);
			

	}

	__global__ void printArray(float* arrayToPrint, const uint32_t size) {
		
		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		if (id == 0) {

			for (uint32_t i = 0; i < size; i++) {
				printf("%.6f ", arrayToPrint[i]);
			}
			printf("\n");
		}
	}

	void usePrintArrayFromCppFile(float* arrayToPrint, const uint32_t size) {

		GpuHelperFunc::printArray << <1, 1 >> > (arrayToPrint, size);
		CHECK_FOR_KERNEL_ERRORS("GpuHelperFunc::usePrintArrayFromCppFile");
	}

	__global__ void printArray_2d(float* arrayToPrint, const uint32_t xsize, const uint32_t ysize) {

		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		if (id == 0) {
			uint32_t i = 0;
			for (uint32_t x = 0; i < xsize; x++) {
				for (uint32_t y = 0; y < ysize; y++) {
					printf("%.6f ", arrayToPrint[i]);
					i++;
				}
				printf("\n");
			}
			printf("\n");
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


		return;


	}

	__global__ void multiplyBy2AndSub1(float* arrayToChange, const uint32_t size) {

		uint32_t id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		while (id < size) {
			arrayToChange[id] = 2 * arrayToChange[id] - 1;
			id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
		}



	}


};