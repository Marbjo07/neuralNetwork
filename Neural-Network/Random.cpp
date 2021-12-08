#pragma once

#include "Random.hpp"




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

    float Range(float x) {
        return Default() * x - x;
    }
};
