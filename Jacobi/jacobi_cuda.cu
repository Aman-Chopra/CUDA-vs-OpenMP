#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <cuda.h>

// Helper function to print the numbers in sorted order
void print_ans(float *arr, int n)
{
    register int i = 0;
    while (i < n)
    {
        printf("%f ", arr[i]);
        i++;
    }
    printf("\n");
}

// Kernel function
__global__ void solve(unsigned long N, float *coefficients, float *oldX, float *newX)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= 0 && index < N)
    {
        newX[index] = 0.0;
        register int j = index * (N + 1);
        while (j < index * (N + 1) + N)
        {
            newX[index] -= (coefficients[j] * oldX[j % (N + 1)]);
            ++j;
        }
        newX[index] += (coefficients[index * (N + 1) + index] * oldX[index]);
        newX[index] += coefficients[index * (N + 1) + N];
        newX[index] /= coefficients[index * (N + 1) + index];
    }
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./jacobi_cuda name\n");
        printf("name = The name of the input file\n");
        exit(1);
    }

    FILE *fp;

    // Opening the file in the read mode
    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("File does not exist \n");
        exit(1);
    }

    // Read the file size
    struct stat st;
    stat(argv[1], &st);
    char *file_str = (char *)malloc(st.st_size);
    fread(file_str, st.st_size, sizeof(char), fp);
    char *rest = file_str;
    char *token = strtok_r(rest, " \t\n", &rest);

    // Read the number of unkowns
    unsigned long N = (unsigned long)atoi(token);

    size_t size = N * sizeof(float);

    // Read the absolute relative error
    token = strtok_r(rest, " \t\n", &rest);
    float e = atof(token);

    // Read initial values
    float *oldX = (float *)malloc(size);

    register int i = 0;
    while (i < N)
    {
        token = strtok_r(rest, " \t\n", &rest);
        oldX[i] = atof(token);
        ++i;
    }

    // Read coefficients and constants
    float *coef = (float *)malloc(N * (N + 1) * sizeof(float));
    i = 0;
    while (i < N * (N + 1))
    {
        token = strtok_r(rest, " \t\n", &rest);
        coef[i] = atof(token);
        i++;
    }

    // Closing the file
    free(file_str);
    fclose(fp);

    int blockSize = 1024;

    // Allocating the device variables
    float *oldX_device;
    cudaMalloc((void **)&oldX_device, size);
    if (!oldX_device)
    {
        printf("Error allocating array oldX_device\n");
        exit(1);
    }

    float *newX_device;
    cudaMalloc((void **)&newX_device, N * sizeof(float));
    if (!newX_device)
    {
        printf("Error allocating array newX_device\n");
        exit(1);
    }

    float *coef_device;
    cudaMalloc((void **)&coef_device, N * (N + 1) * sizeof(float));
    if (!coef_device)
    {
        printf("Error allocating array coef_device\n");
        exit(1);
    }

    // Copying variables from host to the device
    cudaMemcpy(oldX_device, oldX, size, cudaMemcpyHostToDevice);
    cudaMemcpy(newX_device, oldX, size, cudaMemcpyHostToDevice);
    cudaMemcpy(coef_device, coef, N * (N + 1) * sizeof(float), cudaMemcpyHostToDevice);

    float *newX = (float *)malloc(size);
    memcpy(newX, oldX, size);

    int num_iteration = 0;
    int flag = 1;

    // Core logic 
    while (flag > 0)
    {
        num_iteration++;
        solve<<<(N + blockSize - 1) / blockSize, blockSize>>>(N, coef_device, oldX_device, newX_device);
        cudaMemcpy(newX, newX_device, size, cudaMemcpyDeviceToHost);
        flag = 0;
        i = 0;
        while (i < N)
        {
            float error = fabs((newX[i] - oldX[i]) / newX[i]);
            flag += ((int)(error > e));
            i++;
        }
        memcpy(oldX, newX, size);
        cudaMemcpy(oldX_device, newX_device, size, cudaMemcpyDeviceToDevice);
    }

    // Printing the coefficients
    print_ans(newX, N);
    printf("\n");

    // Printing the total number of iterations
    printf("Total number of iterations: %d\n", num_iteration);
    printf("\n");

    // Freeing the host variables
    free(coef);
    free(oldX);
    free(newX);

    // Freeing the device variables
    cudaFree(coef_device);
    cudaFree(newX_device);
    cudaFree(oldX_device);

    return 0;
}