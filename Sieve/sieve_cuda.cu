#include <cuda.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Kernel function declaration
__global__ void sieve(bool *prime_device, int n, int cur_index, int num_threads);

// Helper function to print the prime numbers
void print_ans(bool *primes, int n)
{
    printf("2 ");
    for (int i = 1; i < ((n + 2) / 2); i++)
    {
        if (primes[i] == false && ((2*i) + 1 ) <= n)
        {
            printf("%d ", ((2 * i) + 1));
        }
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./sieve_cuda n\n");
        printf("n = The number till which all the prime numbers are to be found\n");
        exit(1);
    }

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);
    size_t size = sizeof(bool) * ((n + 2) / 2);

    // Allocating memory to the host variable
    bool *primes_host = (bool *)malloc(size);
    if (!primes_host)
    {
        printf("Error allocating array primes_host\n");
        exit(1);
    }

    // Allocating memory to the device variable
    bool *primes_device;
    cudaMalloc((void **)&primes_device, size);
    if (!primes_device)
    {
        printf("Error allocating array primes_device\n");
        exit(1);
    }

    // Initially marking all the device elements of the array as false
    cudaMemset(primes_device, false, size);

    // Extracting each number and letting the kernel mark the multiples of that number as true
    int optimal_threads = 1024;
    register int i = 3;
    while(i <= ceil(sqrt(n)))
    {
        int block = n/i;
        cudaProfilerStart();
        sieve<<<ceil(block/2048.0), optimal_threads>>> (primes_device, n, i, optimal_threads);
        cudaProfilerStop();
        i += 2;
    }

    // Copying the device variable back to the host variable
    cudaMemcpy(primes_host, primes_device, size, cudaMemcpyDeviceToHost);
    
    print_ans(primes_host, n);

    // Free the variables from the device and host memory
    cudaFree(primes_device);
    free(primes_host);

    return 0;
}

// This is the Sieve kernel function
__global__ void sieve(bool *primes_device, int n, int cur_index, int num_threads)
{
    if (!primes_device[cur_index / 2])
    {
        int index = 3 + (2 * ((blockIdx.x * blockDim.x) + threadIdx.x));
        int res_index = index * cur_index;
        if (res_index <= n)
        {
            primes_device[(res_index) / 2] = true;
        }
    }
}

