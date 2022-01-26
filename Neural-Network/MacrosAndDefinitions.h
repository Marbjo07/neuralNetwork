#pragma once

#define SIZEOF(x) sizeof(x) / sizeof(x[0])


// have to be equal
// use cuda kernel functions
#define ACTIVATION_FUNCTION_GPU(x) x / (1 + abs(x))
// use cpu functions
#define ACTIVATION_FUNCTION_CPU(x) x / (1 + abs(x))
// 1 / (1 - x + x*x)

#define GRID_SIZE_NEURALNETWORK 4
#define BLOCK_SIZE_NEURALNETWORK 8


#define CHECK_FOR_KERNEL_ERRORS(identifier) {cudaDeviceSynchronize(); cudaError_t err = cudaGetLastError();if (err != cudaSuccess) {std::cout << "\n" <<cudaGetErrorString(err) <<" by: " << identifier << "\nFILE: " << __FILE__ << " LINE: " << __LINE__ << std::endl;throw std::runtime_error(cudaGetErrorString(err));}}

// used in Test::runTests()
#define ONETEST(NAME, FUNCTION) {int output = Private::FUNCTION(debug); std::cout << NAME << Private::passNotPass(output); if (output != 0 && exitOnFail) { return; }}