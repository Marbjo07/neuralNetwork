#pragma once

#include <iostream>

#ifndef RANDOM_HPP
#define RANDOM_HPP

namespace Random {
    static uint_fast32_t x = 123456789;
    static uint_fast32_t y = 362436069;
    static uint_fast32_t z = 521288629;


    // returns value between -1 and 1
    float Default();

    // return value between -x and x 
    float Range(float x); 
};
#endif // !RANDOM_HPP