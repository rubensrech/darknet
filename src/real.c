#include "real.h"
#include "darknet.h"

#if REAL == HALF

void float2real_array(float* src, real* dst, size_t n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}


void real2float_array(real* src, float* dst, size_t n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}

#endif

float* cast_array_real2float(real *src, int n) {
    float *dst = (float*)malloc(n * sizeof(float));

    #if REAL == FLOAT
        return src;
    #else
        #ifdef GPU
            real *src_gpu = cuda_make_array(src, n);
            float *dst_gpu = cuda_make_float_array(dst, n);
            real2float_array_gpu(src_gpu, dst_gpu, n);
            cuda_pull_float_array(dst_gpu, dst, n);
        #else
            real2float_array(src, dst, n);
        #endif
    #endif

    return dst;
}

real* cast_array_float2real(float *src, int n) {
    real *dst = (real*)malloc(n * sizeof(real));

    #if REAL == FLOAT
        return src;
    #else
        #ifdef GPU
            float *src_gpu = cuda_make_float_array(src, n);
            real *dst_gpu = cuda_make_array(dst, n);
            float2real_array_gpu(src_gpu, dst_gpu, n);
            cuda_pull_array(dst_gpu, dst, n);
        #else
            float2real_array(src, dst, n);
        #endif
    #endif

    return dst;
}