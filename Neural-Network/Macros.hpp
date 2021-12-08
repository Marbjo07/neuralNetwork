#pragma once

#ifndef MACROS_H
#define MACROS_H

// returns number of elements in given array
#define SIZEOF(x) sizeof(x) / sizeof(x[0])

#define ACTIVATION_FUNCTION_GPU(x) x
#define ACTIVATION_FUNCTION_CPU(x) x

#define ROUND_UP(x, m) ((x) + (m) - 1 - ((x) + (m) - 1) % (m))

#endif // !MACROS_H