#include "real.h"

#include <cuda_fp16.h>
#include <cuda.h>

#if REAL == HALF

__global__ void float2real_array_kernel(float* src, real_device* dst, int n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		#if REAL == HALF
			dst[i] = __float2half(src[i]);
		#else
			dst[i] = src[i];
		#endif
	}
}

void float2real_array_gpu(float* src, real* dst, int n) {
	float2real_array_kernel<<<cuda_gridsize(n), BLOCK>>>(src, (real_device*)dst, n);
	check_error(cudaPeekAtLastError());
}

__global__ void real2float_array_kernel(real_device* src, float* dst, int n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		#if REAL == HALF
			dst[i] = __half2float(src[i]);
		#else
			dst[i] = src[i];
		#endif
	}	
}

void real2float_array_gpu(real* src, float* dst, int n) {
	real2float_array_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)src, dst, n);
	check_error(cudaPeekAtLastError());
}

#endif