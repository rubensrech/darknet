#ifndef REAL_H
#define REAL_H

#define FLOAT   0
#define DOUBLE  1
#define HALF    2

#if REAL == DOUBLE
    typedef double real;
    #define REAL_MAX __DBL_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_DOUBLE

    #ifdef GPU
        typedef double3 real3;
    #endif
#elif REAL == HALF
    #include <cuda_fp16.h>
    #include "half.hpp"

    typedef half real;
    #define REAL_MAX real(65504)
    #define CUDNN_DATA_REAL CUDNN_DATA_HALF
    
    #ifdef GPU
        typedef half3 real3;
    #endif
#else
    typedef float real;
    #define REAL_MAX __FLT_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
    
    #ifdef GPU
        typedef float3 real3;
    #endif
#endif

#endif