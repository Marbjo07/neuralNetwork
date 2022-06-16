
#include "GpuHelperFunctions.cuh"

namespace GpuHelperFunc {

	__global__ void setAllElemetnsInArrayToOneVal(float* arrayToChange, const uint32_t size, float val) {

		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
			arrayToChange[id] = val;
		}
	}

	void cublasCompute(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int uiWB, int uiHA, int uiWA, const int deviceNum) {
		cudaSetDevice(deviceNum);
		
		float alpha = 1;
		float beta = 0;
		//       Signature: handel, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
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
				printf("%.3f ", arrayToPrint[i]);
			}
		}
	}

	void usePrintArrayFromCppFile(float* arrayToPrint, const uint32_t size, const int deviceNum, cudaStream_t deviceStream) {

		cudaSetDevice(deviceNum);

		GpuHelperFunc::printArray << <1, 1, 0, deviceStream >> > (arrayToPrint, size);
		CHECK_FOR_KERNEL_ERRORS;
	}

	__global__ void printArray_2d(float* arrayToPrint, const uint32_t xsize, const uint32_t ysize) {

		int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		if (id == 0) {
			uint32_t i = 0;
			for (uint32_t x = 0; i < xsize; x++) {
				for (uint32_t y = 0; y < ysize; y++) {
					printf("%.3f ", arrayToPrint[i]);
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
	}

	__global__ void multiplyBy2AndSub1(float* arrayToChange, const uint32_t size) {

		uint32_t id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		while (id < size) {
			arrayToChange[id] = 2 * arrayToChange[id] - 1;
			id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
		}



	}

	namespace forEach {

		__global__ void add(float* a, const float* b, const float* c, const int size) {
			int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

			for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
				a[id] = b[id] + c[id];
			}
		}
		__global__ void sub(float* a, const float* b, const float* c, const int size) {
			int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

			for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
				a[id] = b[id] - c[id];
			}
		}
		__global__ void mul(float* a, const float* b, const float* c, const int size) {
			int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

			for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
				a[id] = b[id] * c[id];
			}
		}
		__global__ void div(float* a, const float* b, const float* c, const int size) {
			int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

			for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
				a[id] = b[id] / c[id];
			}
		}


		namespace constVal {

			__global__ void add(float* a, const float* b, const float constVal, const int size) {
				int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

				for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
					a[id] = b[id] + constVal;
				}
			}
			__global__ void sub(float* a, const float* b, const float constVal, const int size) {
				int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

				for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
					a[id] = b[id] - constVal;
				}
			}
			__global__ void mul(float* a, const float* b, const float constVal, const int size) {
				int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

				for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
					a[id] = b[id] * constVal;
				}

			}
			__global__ void div(float* a, const float* b, const float constVal, const int size) {
				int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

				for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
					a[id] = b[id] / constVal;
				}
			}
		}

	}


};