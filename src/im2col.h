#ifndef IM2COL_H
#define IM2COL_H

#include "darknet.h"

void im2col_cpu(real* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, real* data_col);

#ifdef GPU

void im2col_gpu(real *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, real *data_col);

void im2col_float_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col);

void im2col_half_gpu(half_host *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, half_host *data_col);

#endif
#endif
