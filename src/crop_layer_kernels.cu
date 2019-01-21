#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
}

__device__ real get_pixel_kernel(real *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ real3 rgb_to_hsv_kernel(real3 rgb)
{
    real r = rgb.x;
    real g = rgb.y; 
    real b = rgb.z;

    real h, s, v;
    real max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    real min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    real delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
#if REAL == DOUBLE
    return make_double3(h, s, v);
#else
    return make_float3(h, s, v);
#endif
}

__device__ real3 hsv_to_rgb_kernel(real3 hsv)
{
    real h = hsv.x;
    real s = hsv.y; 
    real v = hsv.z;

    real r, g, b;
    real f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floorf(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
#if REAL == DOUBLE
    return make_double3(r, g, b);
#else
    return make_float3(r, g, b);
#endif
}

__device__ real bilinear_interpolate_kernel(real *image, int w, int h, real x, real y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    real dx = x - ix;
    real dy = y - iy;

    real val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void levels_image_kernel(real *image, real *rand, int batch, int w, int h, int train, real saturation, real exposure, real translate, real scale, real shift)
{
    int size = batch * w * h;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    real rshift = rand[0];
    real gshift = rand[1];
    real bshift = rand[2];
    real r0 = rand[8*id + 0];
    real r1 = rand[8*id + 1];
    real r2 = rand[8*id + 2];
    real r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;

    size_t offset = id * h * w * 3;
    image += offset;
    real r = image[x + w*(y + h*0)];
    real g = image[x + w*(y + h*1)];
    real b = image[x + w*(y + h*2)];
#if REAL == DOUBLE
    real3 rgb = make_double3(r,g,b);
#else
    real3 rgb = make_float3(r,g,b);
#endif
    if(train){
        real3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5f)*shift;
    image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5f)*shift;
    image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5f)*shift;
}

__global__ void forward_crop_layer_kernel(real *input, real *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, real angle, real *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    real cx = w/2.f;
    real cy = h/2.f;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    real r4 = rand[8*b + 4];
    real r5 = rand[8*b + 5];
    real r6 = rand[8*b + 6];
    real r7 = rand[8*b + 7];

    real dw = (w - crop_width)*r4;
    real dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5f));
    angle = 2*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    real x = (flip) ? w - dw - j - 1 : j + dw;    
    real y = i + dh;

    real rx = cosf(angle)*(x-cx) - sinf(angle)*(y-cy) + cx;
    real ry = sinf(angle)*(x-cx) + cosf(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}

extern "C" void forward_crop_layer_gpu(crop_layer layer, network net)
{
    cuda_random(layer.rand_gpu, layer.batch*8);

    real radians = layer.angle*3.14159265f/180.f;

    real scale = 2;
    real translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
    check_error(cudaPeekAtLastError());

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, radians, layer.output_gpu);
    check_error(cudaPeekAtLastError());

/*
       cuda_pull_array(layer.output_gpu, layer.output, size);
       image im = real_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
       image im2 = real_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
       image im3 = real_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));

       translate_image(im, -translate);
       scale_image(im, 1/scale);
       translate_image(im2, -translate);
       scale_image(im2, 1/scale);
       translate_image(im3, -translate);
       scale_image(im3, 1/scale);
       
       show_image(im, "cropped");
       show_image(im2, "cropped2");
       show_image(im3, "cropped3");
       cvWaitKey(0);
       */
}

