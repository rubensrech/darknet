#include "real.h"
#include "darknet.h"

void float2real_array(float* src, real* dst, int n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}


void real2float_array(real* src, float* dst, int n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}

/*
 *  @param src - CPU array
 *  @param dst_gpu - if NULL, 'src' will be converted to float and the output array will be available in CPU,
 *                   otherwise, if GPU && dst != NULL, the output array will be will be kept in GPU.
 */
float* cast_array_real2float(real *src, int n, float *dst_gpu) {
    #if REAL == FLOAT
        #ifdef GPU
            if (dst_gpu) {
                // Push 'src' to 'dst_gpu'
                cuda_push_array(dst_gpu, src, n);
                return dst_gpu;
            } else {
                return src;
            }
        #else
            return src;
        #endif
    #else
        #ifdef GPU
            real *src_gpu = cuda_make_array(src, n);
            if (dst_gpu) {
                real2float_array_gpu(src_gpu, dst_gpu, n);
                cudaFree(src_gpu);
                return dst_gpu;
            } else {
                dst_gpu = cuda_make_float_array(NULL, n);
                float *dst = (float*)calloc(n, sizeof(float));
                real2float_array_gpu(src_gpu, dst_gpu, n);
                cuda_pull_float_array(dst_gpu, dst, n);
                cudaFree(src_gpu);
                cudaFree(dst_gpu);
                return dst;
            }
        #else
            float *dst = (float*)calloc(n, sizeof(float));
            real2float_array(src, dst, n);
            return dst;
        #endif
    #endif
}


/*
 *  @param src - CPU array
 *  @param dst_gpu - if NULL, 'src' will be converted to 'real' and the output array will be available in CPU,
 *                   otherwise, if GPU && dst != NULL, the output array will be will be kept in GPU.
 */
real* cast_array_float2real(float *src, int n, real *dst_gpu) {
    #if REAL == FLOAT
        #ifdef GPU
            if (dst_gpu) {
                // Push 'src' to 'dst_gpu'
                cuda_push_array(dst_gpu, src, n);
                return dst_gpu;
            } else {
                return src;
            }
        #else
            return src;
        #endif
    #else
        #ifdef GPU
            float *src_gpu = cuda_make_float_array(src, n);
            if (dst_gpu) {
                float2real_array_gpu(src_gpu, dst_gpu, n);
                cudaFree(src_gpu);
                return dst_gpu;
            } else {
                dst_gpu = cuda_make_array(NULL, n);
                real *dst = (real*)calloc(n, sizeof(real));
                float2real_array_gpu(src_gpu, dst_gpu, n);
                cuda_pull_array(dst_gpu, dst, n);
                cudaFree(src_gpu);
                cudaFree(dst_gpu);
                return dst;
            }
        #else
            real *dst = (real*)calloc(n, sizeof(real));
            float2real_array(src, dst, n);
            return dst;
        #endif
    #endif
}