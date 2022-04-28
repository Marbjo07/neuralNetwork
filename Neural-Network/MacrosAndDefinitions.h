#pragma once

#define SIZEOF(x) sizeof(x) / sizeof(x[0])


// have to be equal
// use cuda kernel functions
#define ACTIVATION_FUNCTION_GPU(x) x
// use cpu functions
#define ACTIVATION_FUNCTION_CPU(x) x


#define GRID_SIZE_NEURALNETWORK 1
#define BLOCK_SIZE_NEURALNETWORK 1


#define CHECK_FOR_KERNEL_ERRORS(identifier)                           \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)



// used in Test::runTests()
#define ONETEST(NAME, FUNCTION) {int output = Private::FUNCTION(debug); std::cout << NAME << Private::passNotPass(output); if (output != 0 && exitOnFail) { return; }}