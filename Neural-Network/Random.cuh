#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#include "MacrosAndDefinitions.h"
#include "GpuHelperFunctions.cuh"

#ifndef RANDOM_HPP
#define RANDOM_HPP

namespace Random {

    // returns value between -1 and 1
    float Default();

    void MutateArray(float* arrayToRandomize, curandGenerator_t* randomNumberGen, const int size);

    __global__ void MatrixMul_1d(float* x1, float* x2, const int size);

    // return value between -x and x 
    float Range(float x); 
};
#endif // !RANDOM_HPP