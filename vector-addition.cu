#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

void vecAddCPU(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecAddKernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int size = 12345678;
    const int threadsPerBlock = 512;
    const int blocksPerGrid = ceil(static_cast<float>(size) / threadsPerBlock);

    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C_CPU = new float[size];
    float* h_C_GPU = new float[size];

    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    clock_t start_cpu = clock();
    vecAddCPU(h_A, h_B, h_C_CPU, size);
    clock_t end_cpu = clock();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    std::cout << "Vector size " << size << "\n\n";
    std::cout << "vecAdd on CPU: " << cpu_time << "s\n";

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    clock_t start_cuda_malloc = clock();
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);
    clock_t end_cuda_malloc = clock();
    double cuda_malloc_time = static_cast<double>(end_cuda_malloc - start_cuda_malloc) / CLOCKS_PER_SEC;
    std::cout << "cudaMalloc: " << cuda_malloc_time << "s\n";

    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    clock_t start_kernel = clock();
    vecAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    clock_t end_kernel = clock();
    double kernel_time = static_cast<double>(end_kernel - start_kernel) / CLOCKS_PER_SEC;
    std::cout << "vecAddKernel<<<(" << blocksPerGrid << ",1,1),(" << threadsPerBlock << ",1,1)>>>: " << kernel_time << "s\n";

    clock_t start_cuda_memcpy = clock();
    cudaMemcpy(h_C_GPU, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end_cuda_memcpy = clock();
    double cuda_memcpy_time = static_cast<double>(end_cuda_memcpy - start_cuda_memcpy) / CLOCKS_PER_SEC;
    std::cout << "cudaMemcpy: " << cuda_memcpy_time << "s\n";
    std::cout << "vecAdd on GPU: " << kernel_time + cuda_memcpy_time << "s\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_GPU;

    return 0;
}
