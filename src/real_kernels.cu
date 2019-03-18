#include "real.h"

#include <cuda_fp16.h>
#include <cuda.h>

#if REAL == HALF

__global__ void float2half_array_kernel(float* src, __half* dst, size_t n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = __float2half(src[i]);
}

void float2half_array_gpu(float* src, real* dst, size_t n) {
	float2half_array_kernel<<<cuda_gridsize(n), BLOCK>>>(src, (__half*)dst, n);
	check_error(cudaPeekAtLastError());
}

__global__ void half2float_array_kernel(__half* src, float* dst, size_t n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = __half2float(src[i]);
}

void half2float_array_gpu(real* src, float* dst, size_t n) {
	half2float_array_kernel<<<cuda_gridsize(n), BLOCK>>>(src, (__half*)dst, n);
	check_error(cudaPeekAtLastError());
}

#endif