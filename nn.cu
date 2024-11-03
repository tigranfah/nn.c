#include <stdio.h>

// Define the CUDA kernel function
__global__ void helloWorldKernel() {
    // Identify the thread ID within the grid
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", threadId);
}

int main() {
    // Define the number of blocks and threads per block
    int numBlocks = 1;
    int threadsPerBlock = 10;

    // Launch the kernel
    helloWorldKernel<<<numBlocks, threadsPerBlock>>>();

    // Synchronize to make sure all threads finish before the program exits
    cudaDeviceSynchronize();

    return 0;
}