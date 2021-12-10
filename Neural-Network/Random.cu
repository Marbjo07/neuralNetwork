#pragma once

#include "Random.cuh"




namespace Random {

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

    __global__ void ArrayGpu(float* arrayToRandomize, const int size, uint_fast32_t d_x, uint_fast32_t d_y, uint_fast32_t d_z) {

        int id = ((gridDim.x * blockIdx.y) + blockIdx.x * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        unsigned long t;
        uint_fast32_t x = d_x + id * id;
        uint_fast32_t y = d_y + id * id;
        uint_fast32_t z = d_z + id * id;


        while (id < size) {
            
            x ^= x << 16;
            x ^= x >> 5;
            x ^= x << 1;

            t = x;
            x = y;
            y = z;
            z = t ^ x ^ y;

            arrayToRandomize[id] = (float)z / 2147483648 - 1;

            id += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
        }

        __syncthreads();

        if (id == 1) {
            d_x = x;
            d_y = y;
            d_z = z;
        }
    }

    float Range(float x) {
        return Default() * x - x;
    }
};
