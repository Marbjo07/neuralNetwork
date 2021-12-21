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

    __global__ void ArrayGpu(float* arrayToRandomize, const int size, int64_t offset) {

        int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

        uint32_t x = id + 1 + offset;

        // cycle a few times or else the output are close together
        for (auto i = 0; i < 5; i++) {
            x ^= (x << 17);
            x ^= (x >> 13);
            x ^= (x << 5);
        }

        while (id < size) {
            x ^= (x << 17);
            x ^= (x >> 13);
            x ^= (x << 5);

            arrayToRandomize[id] = (float)x / 4294967296 - 1;

            id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
        }

        __syncthreads();

        if (id == 0) {
            offset = x;
        }

        return;
    }
    __global__ void MutateArrayGpu(float* arrayToRandomize, const int size, int64_t offset) {
        int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        

        uint32_t x = id + 1 + offset;

        // numbers can look alike if this isnt done
        for (auto i = 0; i < 10; i++) {
            x ^= (x << 17);
            x ^= (x >> 13);
            x ^= (x << 5);
        }

        while (id < size) {

            x ^= (x << 17);
            x ^= (x >> 13);
            x ^= (x << 5);

            if (std::abs(arrayToRandomize[id]) <= 0.0001) {
                arrayToRandomize[id] += (float)x / 4294967296 - 1;
            }
            arrayToRandomize[id] *= (float)x / 4294967296 - 1;
            id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
        }

        __syncthreads();        
        
        if (id == 0) {
            offset = x;
        }
    }

    float Range(float x) {
        return Default() * x - x;
    }
};
