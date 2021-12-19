#pragma once

#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef RANDOM_HPP
#define RANDOM_HPP

namespace Random {
    static uint_fast32_t offset = 362436070;

    // returns value between -1 and 1
    float Default();

    __global__ void ArrayGpu(float* arrayToRandomize, const int size, int offset);

    __global__ void MutateArrayGpu(float* arrayToRandomize, const int size, int offset);

    // return value between -x and x 
    float Range(float x); 
};
#endif // !RANDOM_HPP