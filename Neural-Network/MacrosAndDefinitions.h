#pragma once

#define GRID_SIZE_NEURALNETWORK 4
#define BLOCK_SIZE_NEURALNETWORK 32


#define CHECK_FOR_KERNEL_ERRORS                                       \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)



// used in Test::runTests()
#define ONETEST(NAME, FUNCTION) {int output = Private::FUNCTION(debug); std::cout << NAME << Private::passNotPass(output); if (output != 0 && exitOnFail) { return; }}