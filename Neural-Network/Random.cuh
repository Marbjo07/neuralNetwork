#pragma once

#include <iostream>
#include "E:\CUDA\Cuda Development\include\cuda.h"
#include "E:\CUDA\Cuda Development\include\cuda_runtime.h"
#include "E:\CUDA\Cuda Development\include\device_launch_parameters.h"

#ifndef RANDOM_HPP
#define RANDOM_HPP

namespace Random {
    static uint_fast32_t x = 123456789;
    static uint_fast32_t y = 362436069;
    static uint_fast32_t z = 521288629;

    static uint_fast32_t d_x = 123456789;
    static uint_fast32_t d_y = 362436069;
    static uint_fast32_t d_z = 521288630;

    // returns value between -1 and 1
    float Default();

    __global__ void ArrayGpu(float* arrayToRandomize, const int size, uint_fast32_t d_x, uint_fast32_t d_y, uint_fast32_t d_z);

    // return value between -x and x 
    float Range(float x); 
};
#endif // !RANDOM_HPP