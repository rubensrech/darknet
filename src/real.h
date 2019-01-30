#ifndef REAL_H
#define REAL_H

#define FLOAT   0
#define DOUBLE  1
#define HALF    2

#if REAL == DOUBLE
    typedef double real;
    typedef double real_device;
    #define REAL_MAX __DBL_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_DOUBLE

#elif REAL == HALF
    #ifdef GPU
        #include <cuda_fp16.h>
    #endif
    
    #include "half.hpp"
    using half_float::half;
    using namespace half_float::literal;

    typedef half_float::half real;
    typedef __half real_device;
    #define REAL_MAX half(65504)
    #define CUDNN_DATA_REAL CUDNN_DATA_HALF

#else
    typedef float real;
    typedef float real_device;
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

#endif