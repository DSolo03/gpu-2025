#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <iostream>
#include <stdexcept>

__global__ void normalize_kernel(float2 *data, int n, int batch);

void cleanup_fft(cufftComplex *d_input, cufftComplex *d_fft,
                 cudaStream_t stream);

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    if (input.empty() || batch <= 0)
        throw std::invalid_argument("Invalid input");

    int total_elements = input.size();
    if (total_elements % (2 * batch) != 0)
        throw std::invalid_argument("Invalid size");

    int n = total_elements / (2 * batch);
    std::vector<float> output(total_elements);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cufftComplex *d_input = nullptr, *d_fft = nullptr;
    size_t data_size = total_elements * sizeof(float);

    cudaMallocAsync(&d_input, data_size, stream);
    cudaMallocAsync(&d_fft, data_size, stream);

    cudaMemcpyAsync(d_input, input.data(), data_size,
                    cudaMemcpyHostToDevice, stream);

    cufftHandle plan_forward, plan_inverse;

    cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch);
    cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch);

    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);

    cufftExecC2C(plan_forward, d_input, d_fft, CUFFT_FORWARD);
    cufftExecC2C(plan_inverse, d_fft, d_input, CUFFT_INVERSE);

    int threads = 256;
    int blocks = (n * batch + threads - 1) / threads;

    normalize_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2 *>(d_input), n, batch);

    cudaMemcpyAsync(output.data(), d_input, data_size,
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cleanup_fft(d_input, d_fft, stream);

    return output;
}

__global__ void normalize_kernel(float2 *data, int n, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * batch;

    if (idx < total)
    {
        float scale = 1.0f / n;
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

void cleanup_fft(cufftComplex *d_input, cufftComplex *d_fft,
                 cudaStream_t stream)
{
    if (d_input) cudaFreeAsync(d_input, stream);
    if (d_fft) cudaFreeAsync(d_fft, stream);
    if (stream) cudaStreamDestroy(stream);
}