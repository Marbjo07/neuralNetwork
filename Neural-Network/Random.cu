#pragma once

#include "Random.cuh"



namespace Random {


    static uint_fast32_t x = 123456789;
    static uint_fast32_t y = 362436069;
    static uint_fast32_t z = 521288629;



    // return value between -1 and 1
    float Default() {

        unsigned long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        // does this to get a random float between -1 and 1
        return (float)z / 2147483648 - 1;
    }

    __global__ void MatrixMul_1d(float* x1, float* x2, const int size) {
        
        int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#pragma unroll
        for (; id < size; id += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
            x1[id] *= x2[id];
        }

    }

    void MutateArray(float* arrayToRandomize, curandGenerator_t* randomNumberGen, const int size) {

        float* randomNumbers;
        cudaMalloc(&randomNumbers, size * sizeof(float));

        curandGenerateUniform((*randomNumberGen), randomNumbers, size);

        CHECK_FOR_KERNEL_ERRORS("Random::MutateArray()");


        dim3 DimGrid(2, 2, 1);
        dim3 DimBlock(32, 32, 1);

        MatrixMul_1d << <DimGrid, DimBlock >> > (arrayToRandomize, randomNumbers, size);

        CHECK_FOR_KERNEL_ERRORS("Random::MatrixMul_1d()");
    }

    float Range(float x) {
        return Default() * x - x;
    }
};