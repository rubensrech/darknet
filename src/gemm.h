#ifndef GEMM_H
#define GEMM_H

#include "darknet.h"

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        real *B, int ldb,
        real *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, real ALPHA, 
                    real *A, int lda, 
                    real *B, int ldb,
                    real BETA,
                    real *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, real ALPHA, 
        real *A, int lda, 
        real *B, int ldb,
        real BETA,
        real *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, real ALPHA, 
        real *A_gpu, int lda, 
        real *B_gpu, int ldb,
        real BETA,
        real *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, real ALPHA, 
        real *A, int lda, 
        real *B, int ldb,
        real BETA,
        real *C, int ldc);
#endif
#endif
