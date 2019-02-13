#ifndef REAL_H
#define REAL_H

#define FLOAT       0
#define DOUBLE      1
#define HALF        2

#define CAST(v)     real(v)
#define CAST_DEV(v) real_device(v)

#if REAL == DOUBLE
    typedef double real;
    typedef double real_device;
    #define REAL_MAX __DBL_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_DOUBLE

#elif REAL == HALF
    #ifdef GPU
        #include <cuda_fp16.h>
        typedef __half real_device;
    #endif

    #define HALF_ROUND_STYLE 1  // 1: nearest, -1: truncate (fastest, default)
    #include "half.hpp"
    using half_float::half;
    using namespace half_float::literal;
    typedef half_float::half real;
    
    #define REAL_MAX CAST(65504)
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

#if REAL == HALF
void float2half_array(float* src, real* dst, size_t n);
half habs(half a);
#endif

/* Math functions */

#ifdef __NVCC__

__device__ __forceinline__ real_device exp_real(real_device x) {
#if REAL == HALF
	return hexp(x);
#elif REAL == FLOAT
	return expf(x);
#elif REAL == DOUBLE
	return exp(x);
#endif
}

__device__ __forceinline__ real_device floor_real(real_device x) {
#if REAL == HALF
	return hfloor(x);
#elif REAL == FLOAT
	return floorf(x);
#elif REAL == DOUBLE
	return floor(x);
#endif
}

#endif

#endif