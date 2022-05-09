#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

// Helper function to print the numbers in sorted order
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

// Quick sort helper function to parallelise the two subarrays
void helper(int *arr, int left, int right)
{
    int index = left + ((right - left) / 2);
    int pivot = arr[index];
    int left_value;
    int right_value;
    int *left_ptr = arr + left;
    int *right_ptr = arr + right;
    
    while (left_ptr <= right_ptr)
    {
        left_value = *left_ptr;
        right_value = *right_ptr;

        // Move elements smaller than the pivot value to the left subarray
        for(;left_value < pivot && left_ptr < arr + right;)
        {
            left_ptr++;
            left_value = *left_ptr;
        }

        // Move elements larger than the pivot value to the right subarray
        for(;right_value > pivot && right_ptr > arr + left;)
        {
            right_ptr--;
            right_value = *right_ptr;
        }

        if (left_ptr <= right_ptr)
        {
            *left_ptr = right_value;
            *right_ptr = left_value;
            left_ptr++;
            right_ptr--;
        }
    }

    int new_right = right_ptr - arr;
    int new_left = left_ptr - arr;

    // Launch a new block to sort the right and the left parts
    if (left < new_right)
    {
#pragma omp task
        {
            helper(arr, left, new_right);
        }
    }

    if (new_left < right)
    {
#pragma omp task
        {
            helper(arr, new_left, right);
        }
    }
}

// Quick sort function which calls the helper function
void omp_quick_sort(int *arr, int n)
{
    int num_threads = 10;
    omp_set_num_threads(num_threads);
#pragma omp target data map(tofrom : arr) map(to : n)
#pragma omp parallel
    {
#pragma omp single nowait
        {
            helper(arr, 0, n - 1);
        }
    }
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./quicksort_openmp name\n");
        printf("name = The name of the input file\n");
        exit(1);
    }

    int n;
    FILE *fp;

    // Opening the file in the read mode
    fp = fopen(argv[1], "r");
    if(fp == NULL)
    {
        printf("File does not exist \n");
        exit(1);
    }

    // Getting the number of elements in the file
    fscanf(fp, "%d", &n);

    // Allocating the array to input all the elements from the file
    size_t size = n * sizeof(int);
    int *input = (int *)malloc(size);

    register int i = 0;

    // Populating the array
    while(i < n)
    {
        fscanf(fp, "%d", &input[i]);
        i++;
    }

    // Core logic
    omp_quick_sort(input, n);

    // Printing the output
    print_ans(input, n);

    return 0;
}
