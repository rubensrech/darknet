#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "activations.h"
#include "cuda.h"

// > Activate kernels

template<typename T>
__device__ T linear_activate_kernel(T x) {
    return x;
}

template<typename T>
__device__ T logistic_activate_kernel(T x) {
    return 1 / (1 + exp(-x));
}
__device__ __half logistic_activate_kernel(__half x) {
    return __half(1) / (__half(1) + hexp(-x));
}

template<typename T>
__device__ T loggy_activate_kernel(T x) {
    return 2. / (1. + exp(-x)) - 1;
}
__device__ __half loggy_activate_kernel(__half x) {
    return __half(2.) / (__half(1.) + hexp(-x)) - __half(1);
}

template<typename T>
__device__ T relu_activate_kernel(T x) {
    return x * T(x > T(0));
}

template<typename T>
__device__ T elu_activate_kernel(T x) {
    return (x >= 0)*x + (x < 0) * (exp(x) - 1);
}
__device__ __half elu_activate_kernel(__half x) {
    return __half(x >= __half(0))*x + __half(x < __half(0)) * (hexp(x) - __half(1));
}

template<typename T>
__device__ T selu_activate_kernel(T x) {
    return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f * (exp(x) - 1);
}
__device__ __half selu_activate_kernel(__half x) {
    return __half(x >= __half(0)) * __half(1.0507f) * x + __half(x < __half(0))*__half(1.0507f*1.6732f) * (hexp(x) - __half(1));
}

template<typename T>
__device__ T relie_activate_kernel(T x) {
    return (x > T(0)) ? x : T(.01)*x;
}

template<typename T>
__device__ T ramp_activate_kernel(T x) {
    return x*T(x > T(0)) + T(.1)*x;
}

template<typename T>
__device__ T leaky_activate_kernel(T x) {
    return (x > T(0)) ? x : T(.1)*x;
}

template<typename T>
__device__ T tanh_activate_kernel(T x) {
    return (2.0 / (1 + exp(-2*x)) - 1);
}
__device__ __half tanh_activate_kernel(__half x) {
    return (__half(2) / (__half(1) + hexp(__half(-2)*x)) - __half(1));
}

template<typename T>
__device__ T plse_activate_kernel(T x) {
    if(x < T(-4)) return T(.01f) * (x + T(4));
    if(x > T(4))  return T(.01f) * (x - T(4)) + T(1);
    return T(.125)*x + T(.5);
}

template<typename T>
__device__ T stair_activate_kernel(T x) {
    int n = floor(x);
    if (n % 2 == 0) return floor(x / 2);
    else return (x - n) + floor(x / 2);
}
__device__ __half stair_activate_kernel(__half x) {
    int n = hfloor(x);
    if (n % 2 == 0) return hfloor(x / __half(2));
    else return (x - __half(n)) + hfloor(x / __half(2));
}

template<typename T>
__device__ T hardtan_activate_kernel(T x) {
    if (x < T(-1)) return T(-1);
    if (x > T(1)) return T(1);
    return x;
}

template<typename T>
__device__ T lhtan_activate_kernel(T x) {
    if(x < T(0)) return T(.001)*x;
    if(x > T(1)) return T(.001) * (x - T(1.0)) + T(1.0);
    return x;
}


template<typename T>
__device__ T activate_kernel(T x, ACTIVATION a) {
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

template<typename T>
__global__ void activate_array_kernel(T *x, int n, ACTIVATION a) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

void activate_array_gpu(float *x, int n, ACTIVATION a) {
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    check_error(cudaPeekAtLastError());
}
void activate_array_gpu(double *x, int n, ACTIVATION a) {
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    check_error(cudaPeekAtLastError());
}
void activate_array_gpu(half_float::half *x, int n, ACTIVATION a) {
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>((__half*)x, n, a);
    check_error(cudaPeekAtLastError());
}

// > Gradient kernels

template<typename T>
__device__ T linear_gradient_kernel(T x) {
    return 1;
}

template<typename T>
__device__ T logistic_gradient_kernel(T x) {
    return (T(1)-x)*x;
}

template<typename T>
__device__ T loggy_gradient_kernel(T x) {
    T y = (x + T(1)) / T(2);
    return T(2)*(T(1)-y)*y;
}

template<typename T>
__device__ T relu_gradient_kernel(T x) {
    return T(x > T(0));
}

template<typename T>
__device__ T elu_gradient_kernel(T x) {
    return T(x >= T(0)) + T(x < T(0))*(x + T(1));
}

template<typename T>
__device__ T selu_gradient_kernel(T x) {
    return T(x >= T(0))*T(1.0507) + T(x < T(0))*(x + T(1.0507*1.6732));
}

template<typename T>
__device__ T relie_gradient_kernel(T x) {
    return (x > T(0)) ? 1 : .01f;
}

template<typename T>
__device__ T ramp_gradient_kernel(T x) {
    return T(x > T(0)) + T(.1f);
}

template<typename T>
__device__ T leaky_gradient_kernel(T x) {
    return (x > T(0)) ? 1 : .1f;
}

template<typename T>
__device__ T tanh_gradient_kernel(T x) {
    return T(1)-x*x;
}

template<typename T>
__device__ T plse_gradient_kernel(T x) {
    return ((x < T(0)) || (x > T(1))) ? .01f : .125f;
}

template<typename T>
__device__ T stair_gradient_kernel(T x) {
    if (floor(x) == x) return 0;
    return 1;
}
__device__ __half stair_gradient_kernel(__half x) {
    if (hfloor(x) == x) return 0;
    return 1;
}

template<typename T>
__device__ T hardtan_gradient_kernel(T x) {
    if ((x > T(-1)) && (x < T(1))) return 1;
    return 0;
}

template<typename T>
__device__ T lhtan_gradient_kernel(T x) {
    if((x > T(0)) && (x < T(1))) return 1;
    return .001;
}


template<typename T>
__device__ T gradient_kernel(T x, ACTIVATION a) {
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

template<typename T>
__global__ void gradient_array_kernel(T *x, int n, ACTIVATION a, T *delta) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}

void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta) {
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
    check_error(cudaPeekAtLastError());
}
void gradient_array_gpu(double *x, int n, ACTIVATION a, double *delta) {
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
    check_error(cudaPeekAtLastError());
}
void gradient_array_gpu(half_float::half *x, int n, ACTIVATION a, half_float::half *delta) {
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>((__half*)x, n, a, (__half*)delta);
    check_error(cudaPeekAtLastError());
}

// ---- x ----

__global__ void binary_gradient_array_kernel(real_device *x, real_device *dy, int n, int s, BINARY_ACTIVATION a, real_device *dx) {
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

void binary_gradient_array_gpu(real *x, real *dx, int n, int size, BINARY_ACTIVATION a, real *y) {
    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>((real_device*)x, (real_device*)dx, n/2, size, a, (real_device*)y);
    check_error(cudaPeekAtLastError());
}
__global__ void binary_activate_array_kernel(real_device *x, int n, int s, BINARY_ACTIVATION a, real_device *y) {
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    real_device x1 = x[b*s + i];
    real_device x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}

void binary_activate_array_gpu(real *x, int n, int size, BINARY_ACTIVATION a, real *y) {
    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>((real_device*)x, n/2, size, a, (real_device*)y);
    check_error(cudaPeekAtLastError());
}