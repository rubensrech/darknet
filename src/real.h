#ifndef REAL_H
#define REAL_H

#define FLOAT   0
#define DOUBLE  1
#define HALF    2

#if REAL == DOUBLE
    typedef double real;
    #define REAL_MAX __DBL_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_DOUBLE

#elif REAL == HALF
    #include <cuda_fp16.h>
    #include "half.hpp"

    typedef half real;
    #define REAL_MAX real(65504)
    #define CUDNN_DATA_REAL CUDNN_DATA_HALF

#else
    typedef float real;
    #define REAL_MAX __FLT_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_FLOAT

#endif

/* real3
 * Based on vector_types.h
 */
typedef struct __device_builtin__ {
    real x;
    real y;
    real z;
} real3;

real3 make_real3(real x, real y, real z);

#endif