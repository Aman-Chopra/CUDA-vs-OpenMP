#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

// Helper function to print the prime numbers
void print_ans(bool *primes, int n)
{
    for (int i = 2; i <= n; i++)
    {
        if (primes[i] == false)
        {
            printf("%d ", i);
        }
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./sieve_openmp n\n");
        printf("n = The number till which all the prime numbers are to be found\n");
        exit(1);
    }

    // Variables for loops
    register int i, j;

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);

    // Allocating memory to store the prime numbers
    bool *primes = (bool *)calloc(n+1, sizeof(bool));

    // Mapping the data primes to the device
#pragma omp enter target data map(to : primes [0:n])
    {
        for (i = 2; i <= ceil(sqrt(n)); i++)
        {
            if ((i == 2 || i % 2 != 0) && !primes[i])
            {
                // Make teams and distribute the workload
#pragma omp target
#pragma omp teams distribute parallel for
                for (j = 2 * i; j <= n; j += i)
                {
                    primes[j] = true;
                }

                // Update the primes array from device to host
#pragma omp update from(primes [0:n])
            }
        }
    }

    print_ans(primes, n);
    return 0;
}
