#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

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
        printf("usage: ./sieve_serial n\n");
        printf("n = The number till which all the prime numbers are to be found\n");
        exit(1);
    }

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);

    // Allocating memory to store the prime numbers
    bool *primes = (bool *)calloc(n+1, sizeof(bool));

    // Loop variables
    register int i = 2;
    register int j;

    // Core logic
    while(i <= ceil(sqrt(n)))
    {
        if (!primes[i])
        {
            j = 2 * i;
            for (; j <= n; j += i)
            {
                primes[j] = true;
            }
        }
        i++;
    }

    print_ans(primes, n);
    return 0;
}
