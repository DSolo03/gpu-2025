#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <stdexcept>

void cleanup(float *d_a, float *d_b, float *d_c, cudaStream_t stream, cublasHandle_t handle);

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    std::vector<float> c(n * n, 0.0f);

    if (a.size() != n * n || b.size() != n * n)
        throw std::invalid_argument("Invalid matrix size");

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t matrix_size = n * n * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocAsync(&d_a, matrix_size, stream);
    cudaMallocAsync(&d_b, matrix_size, stream);
    cudaMallocAsync(&d_c, matrix_size, stream);

    cudaMemcpyAsync(d_a, a.data(), matrix_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), matrix_size, cudaMemcpyHostToDevice, stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    cudaMemcpyAsync(c.data(), d_c, matrix_size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cleanup(d_a, d_b, d_c, stream, handle);

    return c;
}

void cleanup(float *d_a, float *d_b, float *d_c, cudaStream_t stream, cublasHandle_t handle)
{
    if (d_a) cudaFreeAsync(d_a, stream);
    if (d_b) cudaFreeAsync(d_b, stream);
    if (d_c) cudaFreeAsync(d_c, stream);
    if (stream) cudaStreamDestroy(stream);
    if (handle) cublasDestroy(handle);
}