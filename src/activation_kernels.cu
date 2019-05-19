#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "activations.h"
#include "cuda.h"

__device__ real_device lhtan_activate_kernel(real_device x)
{
    if(x < CAST_DEV(0)) return CAST_DEV(.001f)*x;
    if(x > CAST_DEV(1)) return CAST_DEV(.001f) * (x - CAST_DEV(1.f)) + CAST_DEV(1.f);
    return x;
}
__device__ real_device lhtan_gradient_kernel(real_device x)
{
    if((x > CAST_DEV(0)) && (x < CAST_DEV(1))) return 1;
    return .001;
}

__device__ real_device hardtan_activate_kernel(real_device x)
{
    if (x < CAST_DEV(-1)) return -1;
    if (x > CAST_DEV(1)) return 1;
    return x;
}
__device__ real_device linear_activate_kernel(real_device x){
    return x;
}
__device__ real_device logistic_activate_kernel(real_device x){
    return CAST_DEV(1.f) / (CAST_DEV(1.f) + exp_real(-x));
}
__device__ real_device loggy_activate_kernel(real_device x){
    return CAST_DEV(2.f) / (CAST_DEV(1.f) + exp_real(-x)) - CAST_DEV(1);
}
__device__ real_device relu_activate_kernel(real_device x){
    return x * CAST_DEV(x > CAST_DEV(0));
}
__device__ real_device elu_activate_kernel(real_device x){
    return CAST_DEV(x >= CAST_DEV(0))*x + CAST_DEV(x < CAST_DEV(0)) * (exp_real(x) - CAST_DEV(1));
}
__device__ real_device selu_activate_kernel(real_device x){
    return CAST_DEV(x >= CAST_DEV(0))*CAST_DEV(1.0507f)*x + CAST_DEV(x < CAST_DEV(0))*CAST_DEV(1.0507f*1.6732f) * (exp_real(x) - CAST_DEV(1));
}
__device__ real_device relie_activate_kernel(real_device x){
    return (x > CAST_DEV(0)) ? x : CAST_DEV(.01f)*x;
}
__device__ real_device ramp_activate_kernel(real_device x){
    return x*CAST_DEV(x > CAST_DEV(0)) + CAST_DEV(.1f)*x;
}
__device__ real_device leaky_activate_kernel(real_device x){
    return (x > CAST_DEV(0)) ? x : CAST_DEV(.1f)*x;
}
__device__ real_device tanh_activate_kernel(real_device x){
    return (CAST_DEV(2.f) / (CAST_DEV(1) + exp_real(CAST_DEV(-2)*x)) - CAST_DEV(1));
}
__device__ real_device plse_activate_kernel(real_device x)
{
    if(x < CAST_DEV(-4)) return CAST_DEV(.01f) * (x + CAST_DEV(4));
    if(x > CAST_DEV(4))  return CAST_DEV(.01f) * (x - CAST_DEV(4)) + CAST_DEV(1);
    return CAST_DEV(.125f)*x + CAST_DEV(.5f);
}
__device__ real_device stair_activate_kernel(real_device x)
{
    int n = floor_real(x);
    if (n % 2 == 0) return floor_real(x / CAST_DEV(2));
    else return (x - CAST_DEV(n)) + floor_real(x / CAST_DEV(2));
}
 

__device__ real_device hardtan_gradient_kernel(real_device x) {
    if ((x > CAST_DEV(-1)) && (x < CAST_DEV(1))) return 1;
    return 0;
}
__device__ real_device linear_gradient_kernel(real_device x){
    return 1;
}
__device__ real_device logistic_gradient_kernel(real_device x){
    return (CAST_DEV(1)-x)*x;
}
__device__ real_device loggy_gradient_kernel(real_device x)
{
    real_device y = (x + CAST_DEV(1)) / CAST_DEV(2);
    return CAST_DEV(2)*(CAST_DEV(1)-y)*y;
}
__device__ real_device relu_gradient_kernel(real_device x){
    return CAST_DEV(x > CAST_DEV(0));
}
__device__ real_device elu_gradient_kernel(real_device x){
    return CAST_DEV(x >= CAST_DEV(0)) + CAST_DEV(x < CAST_DEV(0))*(x + CAST_DEV(1));
}
__device__ real_device selu_gradient_kernel(real_device x){
    return CAST_DEV(x >= CAST_DEV(0))*CAST_DEV(1.0507) + CAST_DEV(x < CAST_DEV(0))*(x + CAST_DEV(1.0507*1.6732));
}
__device__ real_device relie_gradient_kernel(real_device x){
    return (x > CAST_DEV(0)) ? 1 : .01f;
}
__device__ real_device ramp_gradient_kernel(real_device x){
    return CAST_DEV(x > CAST_DEV(0)) + CAST_DEV(.1f);
}
__device__ real_device leaky_gradient_kernel(real_device x){
    return (x > CAST_DEV(0)) ? 1 : .1f;
}
__device__ real_device tanh_gradient_kernel(real_device x){
    return CAST_DEV(1)-x*x;
}
__device__ real_device plse_gradient_kernel(real_device x){
    return ((x < CAST_DEV(0)) || (x > CAST_DEV(1))) ? .01f : .125f;
}
__device__ real_device stair_gradient_kernel(real_device x) {
    if (floor_real(x) == x) return 0;
    return 1;
}

__device__ real_device activate_kernel(real_device x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

__device__ real_device gradient_kernel(real_device x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case SELU:
            return selu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__global__ void binary_gradient_array_kernel(real_device *x, real_device *dy, int n, int s, BINARY_ACTIVATION a, real_device *dx)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    real_device x1 = x[b*s + i];
    real_device x2 = x[b*s + s/2 + i];
    if(id < n) {
        real_device de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s/2 + i] = x1*de; 
    }
}

extern "C" void binary_gradient_array_gpu(real *x, real *dx, int n, int size, BINARY_ACTIVATION a, real *y) 
{
    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>((real_device*)x, (real_device*)dx, n/2, size, a, (real_device*)y);
    check_error(cudaPeekAtLastError());
}
__global__ void binary_activate_array_kernel(real_device *x, int n, int s, BINARY_ACTIVATION a, real_device *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    real_device x1 = x[b*s + i];
    real_device x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}

extern "C" void binary_activate_array_gpu(real *x, int n, int size, BINARY_ACTIVATION a, real *y) 
{
    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>((real_device*)x, n/2, size, a, (real_device*)y);
    check_error(cudaPeekAtLastError());
}

__global__ void activate_array_kernel(real_device *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

__global__ void gradient_array_kernel(real_device *x, int n, ACTIVATION a, real_device *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}

void activate_array_gpu(real *x, int n, ACTIVATION a) 
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)x, n, a);
    check_error(cudaPeekAtLastError());
}

void gradient_array_gpu(real *x, int n, ACTIVATION a, real *delta) 
{
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)x, n, a, (real_device*)delta);
    check_error(cudaPeekAtLastError());
}


/* ----------- Float version ----------- */

__device__ float linear_activate_float_kernel(float x){
    return x;
}
__device__ float logistic_activate_float_kernel(float x){
    return 1.f / (1.f + exp(-x));
}
__device__ float loggy_activate_float_kernel(float x){
    return 2.f / (1.f + exp(-x)) - 1;
}
__device__ float relu_activate_float_kernel(float x){
    return x * (x > 0);
}
__device__ float elu_activate_float_kernel(float x){
    return (x >= 0)*x + (x < 0) * (exp(x) - 1);
}
__device__ float selu_activate_float_kernel(float x){
    return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f * (exp(x) - (1));
}
__device__ float relie_activate_float_kernel(float x){
    return (x > 0) ? x : .01f*x;
}
__device__ float ramp_activate_float_kernel(float x){
    return x*(x > 0) + .1f*x;
}
__device__ float leaky_activate_float_kernel(float x){
    return (x > 0) ? x : .1f*x;
}
__device__ float tanh_activate_float_kernel(float x){
    return (2.f / (1 + exp(-2*x)) - 1);
}
__device__ float plse_activate_float_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
__device__ float stair_activate_float_kernel(float x)
{
    int n = floor(x);
    if (n % 2 == 0) return floor(x / 2);
    else return (x - n) + floor(x / 2);
}
__device__ float hardtan_activate_float_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float lhtan_activate_float_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f * (x - 1.f) + 1.f;
    return x;
}


__device__ float activate_float_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_float_kernel(x);
        case LOGISTIC:
            return logistic_activate_float_kernel(x);
        case LOGGY:
            return loggy_activate_float_kernel(x);
        case RELU:
            return relu_activate_float_kernel(x);
        case ELU:
            return elu_activate_float_kernel(x);
        case SELU:
            return selu_activate_float_kernel(x);
        case RELIE:
            return relie_activate_float_kernel(x);
        case RAMP:
            return ramp_activate_float_kernel(x);
        case LEAKY:
            return leaky_activate_float_kernel(x);
        case TANH:
            return tanh_activate_float_kernel(x);
        case PLSE:
            return plse_activate_float_kernel(x);
        case STAIR:
            return stair_activate_float_kernel(x);
        case HARDTAN:
            return hardtan_activate_float_kernel(x);
        case LHTAN:
            return lhtan_activate_float_kernel(x);
    }
    return 0;
}

__global__ void activate_array_float_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_float_kernel(x[i], a);
}

void activate_array_float_gpu(float *x, int n, ACTIVATION a) 
{
    activate_array_float_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    check_error(cudaPeekAtLastError());
}