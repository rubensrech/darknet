#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "im2col.h"
#include "cuda.h"

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

// > Mixed precision kernel (templated)

template <typename T>
__global__ void im2col_gpu_kernel(int n, T* data_im, int height, int width, int ksize, int pad, int stride, int height_col, int width_col, T *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        T* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const T* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : T(0);
                    
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

// > Mixed precision kernel caller

template <typename T>
void im2col_gpu(T *im, int channels, int height, int width, int ksize, int stride, int pad, T *data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(num_kernels, im, height, width,
            ksize, pad, stride, height_col, width_col, data_col);
}
void im2col_gpu(half_host *im, int channels, int height, int width, int ksize, int stride, int pad, half_host *data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(num_kernels, (half_device*)im, height, width,
            ksize, pad, stride, height_col, width_col, (half_device*)data_col);
}
template void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
template void im2col_gpu(double *im, int channels, int height, int width, int ksize, int stride, int pad, double *data_col);