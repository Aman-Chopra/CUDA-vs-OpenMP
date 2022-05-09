#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <omp.h>

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

int main(int argc, char *argv[])
{
    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./jacobi_serial name\n");
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
    register int j = 0;
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

    float *newX = (float *)malloc(size);
    memcpy(newX, oldX, sizeof(oldX));

    // Core logic
    int flag = 1;
    int num_iteration = 0;
    #pragma omp enter target data map(to:oldX[0:N],newX[0:N],coef[0:N*(N+1)])
    while(flag > 0){
        num_iteration++;
        #pragma omp target
        #pragma omp teams distribute parallel for
        for(int i = 0; i < N; ++i){
            newX[i] = 0.0;
            for(int j = i * (N + 1); j < i * (N + 1) + N; ++j){
                newX[i] -= (coef[j] * oldX[j % (N+1)]);
            }
            newX[i] += (coef[i * (N + 1) + i] * oldX[i]);
            newX[i] += coef[i * (N + 1) + N];
            newX[i] /= coef[i * (N + 1) + i];
        }

        flag = 0;
        #pragma omp target map(tofrom:flag)
        #pragma omp teams distribute parallel for reduction(+:flag)
        for(int i = 0; i < N; ++i){
            float error = fabs((newX[i] - oldX[i])/newX[i]);
            flag += ((int)(error > e));
        }
        memcpy(oldX, newX, N * sizeof(float));
    }
    #pragma omp target exit data map(from:oldX[0:N],newX[0:N],coef[0:N*(N+1)])

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

    return 0;
}