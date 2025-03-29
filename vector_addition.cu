#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// More computationally intensive kernel for stress testing
__global__ void stressKernel(float *data, int numElements, int iterations) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float x = data[i];
        // Perform many floating-point operations to stress the GPU
        for (int iter = 0; iter < iterations; iter++) {
            x = sinf(x) + cosf(x);
            x = sqrtf(fabs(x)) * logf(fabs(x) + 1.0f);
            x = expf(fmodf(x, 2.0f)) + x * x;
            x = tanf(x) / (x * x + 0.1f);
        }
        data[i] = x;
    }
}

int main(int argc, char *argv[]) {
    // Default test parameters
    int durationSeconds = 10;
    int iterationsPerKernel = 100;
    
    // Parse command line arguments if provided
    if (argc > 1) durationSeconds = atoi(argv[1]);
    if (argc > 2) iterationsPerKernel = atoi(argv[2]);
    
    printf("GPU Stress Test starting...\n");
    printf("Duration: %d seconds\n", durationSeconds);
    printf("Computational intensity: %d iterations per element\n", iterationsPerKernel);
    
    // Vector size - using a much larger size for stress testing
    int numElements = 10000000; // 10 million elements
    size_t size = numElements * sizeof(float);
    
    printf("Allocating %.2f MB of memory...\n", size / (1024.0 * 1024.0));
    
    // Allocate and initialize host memory
    float *h_data = (float *)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize with random values
    srand(time(NULL));
    for (int i = 0; i < numElements; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_data;
    cudaError_t error = cudaMalloc((void **)&d_data, size);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(error));
        free(h_data);
        return 1;
    }
    
    // Copy data from host to device
    error = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device: %s\n", cudaGetErrorString(error));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // Setup execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Running with %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start, 0);
    
    // Track elapsed time
    time_t startTime = time(NULL);
    time_t currentTime;
    int totalIterations = 0;
    
    // Run kernels in a loop until duration is reached
    do {
        stressKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, numElements, iterationsPerKernel);
        
        // Check for kernel errors
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(error));
            break;
        }
        
        totalIterations++;
        currentTime = time(NULL);
        
        // Report progress every second
        if (totalIterations % 5 == 0) {
            float elapsedSeconds = difftime(currentTime, startTime);
            printf("Running for %.1f seconds (%d iterations)...\r", elapsedSeconds, totalIterations);
            fflush(stdout);
        }
        
    } while (difftime(currentTime, startTime) < durationSeconds);
    
    // Record stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\nStress test completed after %d iterations\n", totalIterations);
    printf("Total GPU time: %.3f seconds\n", milliseconds / 1000.0);
    
    // Copy result back (just to verify it worked)
    error = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy result back to host: %s\n", cudaGetErrorString(error));
    } else {
        printf("First few values after processing: %.6f %.6f %.6f %.6f %.6f\n", 
               h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);
    cudaDeviceReset();
    
    return 0;
} 