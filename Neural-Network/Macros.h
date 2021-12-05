#pragma once

#ifndef MACROS_H
#define MACROS_H

// returns number of elements in given array
#define SIZEOF(x) sizeof(x) / sizeof(x[0])

#define ACTIVATION_FUNCTION_GPU(x) 1 - 2 / (1+powf(2,2 * x))

#define ACTIVATION_FUNCTION_CPU(x) 1 - 2 / (1+std::pow(2,2 * x))

#define ROUND_UP(x, m) ((x) + (m) - 1 - ((x) + (m) - 1) % (m))

#endif