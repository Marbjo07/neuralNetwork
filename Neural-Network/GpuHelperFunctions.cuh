#include "NeuralNetwork.cuh"

namespace GpuHelperFunc {

	__global__ void setAllElemetnsInArrayToOneVal(float* arrayToChange, const uint32_t size, const float val);

	__global__ void printArray(float* arrayToPrint, const uint32_t size);

	void usePrintArrayFromCppFile(float* arrayToPrint, const uint32_t size, const int deviceNum, cudaStream_t deviceStream);

	__global__ void printArray_2d(float* arrayToPrint, const uint32_t xsize, const uint32_t ysize);

	__global__ void sumOfArray(float* arrayToSum, const uint32_t size, float* sum);

	__global__ void multiplyBy2AndSub1(float* arrayToChange, const uint32_t size);

	void cublasCompute(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int uiWB, int uiHA, int uiWA, const int deviceNum, int _alpha, int _beta);

	namespace forEach {

		__global__ void add(float* a, const float* b, const float* c, const int size);
		__global__ void sub(float* a, const float* b, const float* c, const int size);
		__global__ void mul(float* a, const float* b, const float* c, const int size);
		__global__ void div(float* a, const float* b, const float* c, const int size);


		namespace constVal { 
		
			__global__ void add(float* a, const float* b, const float constVal, const int size);
			__global__ void sub(float* a, const float* b, const float constVal, const int size);
			__global__ void mul(float* a, const float* b, const float constVal, const int size);
			__global__ void div(float* a, const float* b, const float constVal, const int size);
		}

	}

};    