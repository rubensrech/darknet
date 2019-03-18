#include "real.h"

#if REAL == HALF

void float2half_array(float* src, real* dst, size_t n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}


void half2float_array(real* src, float* dst, size_t n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}

float* cast_array_half2float(real *src, int n) {
    float *dst = (float*)malloc(n*sizeof(float));

    #ifdef GPU
        real *src_gpu = cuda_make_array(src, n);
        float *dst_gpu = cuda_make_float_array(dst, n);
        half2float_array_gpu(src_gpu, dst_gpu, n);
        cuda_pull_array(dst_gpu, dst, n);
    #else
        half2float_array(src, dst, n);
    #endif

    return dst;
}

real* cast_array_float2half(float *src, int n) {
    real *dst = (real*)malloc(n*sizeof(real));

    #ifdef GPU
        float *src_gpu = cuda_make_float_array(src, n);
        real *dst_gpu = cuda_make_array(dst, n);
        float2half_array_gpu(src_gpu, dst_gpu, n);
        cuda_pull_array(dst_gpu, dst, n);
    #else
        float2half_array(src, dst, n);
    #endif

    return dst;
}

#endif