#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function to print the addition
void print_ans(int *arr, int n)
{
    register int i = 0;
    while (i < n)
    {
        printf("%d ", arr[i]);
        i++;
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./vector_add_serial n\n");
        printf("n = The size of the vector\n");
        exit(1);
    }

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);

    size_t size = n * sizeof(int);

    // Allocating space for the vectors
    int *vec1 = (int *)malloc(size);
    int *vec2 = (int *)malloc(size);
    int *sum = (int *)malloc(size);

    register int i = 0;

    // Filling the vectors with random values
    while (i < n)
    {
        vec1[i] = rand() % 51;
        vec2[i] = rand() % 51;
        i++;
    }

    // Calculating the sum of the two vectors and storing it in the sum
    i = 0;
    while (i < n)
    {
        sum[i] = vec1[i] + vec2[i];
        i++;
    }

    // Printing the ans
    print_ans(vec1, n);
    print_ans(vec2, n);
    print_ans(sum, n);

    // Freeing the memory
    free(vec1);
    free(vec2);
    free(sum);

    return 0;
}
