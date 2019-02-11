#include "real.h"

#ifdef GPU

#include <cuda_fp16.h>
#include <cuda.h>

__global__ void float2half_array(float* src, __half* dst, size_t n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = __float2half(src[i]);
}

void float2half_array(float* src, real* dst, size_t n) {
	float2half_array<<<cuda_gridsize(n), BLOCK>>>(src, (__half*)dst, n);
	check_error(cudaPeekAtLastError());
}

#endif