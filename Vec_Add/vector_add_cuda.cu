#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

// Helper function to print the addition
void print_ans(int *arr, int n)
{
    register int i = 0;
    while(i < n)
    {
        printf("%d ", arr[i]);
        i++;
    }
    printf("\n");
}

// Kernel function for vector addition
__global__ void vector_add(int n, int *vec1, int *vec2, int *sum)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        sum[index] = vec1[index] + vec2[index];
    }
}

int main(int argc, char *argv[])
{
    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./vector_add_cuda n\n");
        printf("n = The size of the vector\n");
        exit(1);
    }

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);

    // Allocating the host variables
    size_t size = n * sizeof(int);
    int *vec1 = (int *)malloc(size);
    int *vec2 = (int *)malloc(size);
    int *sum = (int *)malloc(size);

    // Allocating the device variables
    int *vec1_device;
    cudaMalloc((void **)&vec1_device, size);
    if (!vec1_device)
    {
        printf("Error allocating array vec1_device\n");
        exit(1);
    }

    int *vec2_device;
    cudaMalloc((void **)&vec2_device, size);
    if (!vec2_device)
    {
        printf("Error allocating array vec2_device\n");
        exit(1);
    }
    
    int *sum_device;
    cudaMalloc((void **)&sum_device, size);
    if (!sum_device)
    {
        printf("Error allocating array sum_device\n");
        exit(1);
    }

    register int i = 0;

    // Initialising the input vectors
    while(i < n)
    {
        vec1[i] = rand() % 51;
        vec2[i] = rand() % 51;
        i++;
    }

    // Copying the vectors to the device
    cudaMemcpy(vec1_device, vec1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_device, vec2, size, cudaMemcpyHostToDevice);

    // Calling the kernel function and copying the answer back to the host
    int optimal_threads = 512;
    dim3 dimGrid((n + 511) / optimal_threads);
    dim3 dimBlock(optimal_threads);
    vector_add<<<dimGrid, dimBlock>>>(n, vec1_device, vec2_device, sum_device);
    cudaMemcpy(sum, sum_device, size, cudaMemcpyDeviceToHost);

    // Printing the ans
    print_ans(vec1, n);
    print_ans(vec2, n);
    print_ans(sum, n);

    // Freeing the device variables
    cudaFree(vec1_device);
    cudaFree(vec2_device);
    cudaFree(sum_device);

    // Freeing the host variables
    free(vec1);
    free(vec2);
    free(sum);

    return 0;
}
