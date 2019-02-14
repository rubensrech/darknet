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

#if REAL == HALF
void float2half_array(float* src, real* dst, size_t n);
half habs(half a);
#endif

#ifdef __NVCC__

/* real3
 * Based on vector_types.h
 */
typedef struct __device_builtin__ {
    real_device x;
    real_device y;
    real_device z;
} real3;

/* Math functions */

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

__device__ __forceinline__ real_device pow_real(real_device x, real_device y) {
#if REAL == HALF
	return CAST_DEV(powf(float(x), float(y)));
#elif REAL == FLOAT
	return powf(x, y);
#elif REAL == DOUBLE
	return pow(x, y);
#endif
}

__device__ __forceinline__ real_device sqrt_real(real_device x) {
#if REAL == HALF
	return hsqrt(x);
#elif REAL == FLOAT
	return sqrtf(x);
#elif REAL == DOUBLE
	return sqrt(x);
#endif
}

__device__ __forceinline__ real_device min_real(real_device x, real_device y) {
#if REAL == HALF
	return fminf(x, y);
#elif REAL == FLOAT
	return fminf(x, y);
#elif REAL == DOUBLE
	return fmin(x, y);
#endif
}

__device__ __forceinline__ real_device max_real(real_device x, real_device y) {
#if REAL == HALF
	return fmaxf(x, y);
#elif REAL == FLOAT
	return fmaxf(x, y);
#elif REAL == DOUBLE
	return fmax(x, y);
#endif
}

__device__ __forceinline__ real_device fabs_real(real_device x) {
#if REAL == HALF
	return fabsf(x);
#elif REAL == FLOAT
	return fabsf(x);
#elif REAL == DOUBLE
	return fabs(x);
#endif
}

__device__ __forceinline__ real_device log_real(real_device x) {
#if REAL == HALF
	return hlog(x);
#elif REAL == FLOAT
	return logf(x);
#elif REAL == DOUBLE
	return log(x);
#endif
}

__device__ __forceinline__ real_device atomicAdd_real(real_device *x, real_device val) {
#if REAL == HALF
    #if __CUDA_ARCH__ > 700
        return atomicAdd((__half*)x, (__half)val);
    #endif

	__half old = *x;
	*x += val;
	return old;
#else
	return atomicAdd(x, val);
#endif
}

#endif

#endif