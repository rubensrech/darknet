#include "real.h"

#ifdef GPU

#include <cuda_fp16.h>
#include <cuda.h>

#if REAL == HALF
__global__ void float2half_array(float* src, __half* dst, size_t n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = __float2half(src[i]);
}

void float2half_array(float* src, half* dst, size_t n) {
	float2half_array<<<cuda_gridsize(n), BLOCK>>>(src, (__half*)dst, n);
	check_error(cudaPeekAtLastError());
}

/* __habs
 * Half absolute on device code
 */
__device__ __half __habs(__half a) {
    uint16_t mask = 0x7fff;
    uint16_t* tmp_a = reinterpret_cast<uint16_t*>(&a);
    *tmp_a = *tmp_a & mask;
    return reinterpret_cast<__half&>(*tmp_a);
}

/* habs
 * Half absolute on host code
 */
half habs(half a) {
    uint16_t mask = 0x7fff;
    uint16_t* tmp_a = reinterpret_cast<uint16_t*>(&a);
    *tmp_a = *tmp_a & mask;
    return reinterpret_cast<half&>(*tmp_a);
}
#endif

#endif