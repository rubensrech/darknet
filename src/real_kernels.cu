#include "real.h"

#include <cuda_fp16.h>
#include <cuda.h>

template<typename T1, typename T2>
__global__ void array_cast_kernel(T1* src, T2* dst, int n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		dst[i] = src[i];
	}
}

void half2real_array_gpu(half_host* src, real* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>((half_device*)src, (real_device*)dst, n);
	check_error(cudaPeekAtLastError());
}
void real2half_array_gpu(real* src, half_host* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)src, (half_device*)dst, n);
	check_error(cudaPeekAtLastError());
}



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