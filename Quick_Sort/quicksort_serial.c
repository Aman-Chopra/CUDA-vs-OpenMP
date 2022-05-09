#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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

// Helper function to swap elements
void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

// Pivot calculator
int partition(int* arr, int left, int right) {
  
  int index = left + ((right - left) / 2);
  swap(&arr[index], &arr[right]);
  int pivot = arr[right];
  
  // Pointer for greater element
  int i = (left - 1);
  int j = left;

  // Compare all elements of the array with the pivot
  while (j < right) {
    if (arr[j] <= pivot) {
      i++;
      swap(&arr[i], &arr[j]);
    }
    j++;
  }

  // Swap pivot to the current index for it to be in middle
  int partition_index = i + 1;
  swap(&arr[partition_index], &arr[right]);
  
  // Return the partition point
  return partition_index;
}

void quicksort_serial(int arr[], int left, int right) {
  if (left < right) {
    
    // Find the pivot element
    int pivot = partition(arr, left, right);
    
    // Recursive call on the left of pivot
    quicksort_serial(arr, left, pivot - 1);
    
    // Recursive call on the right of pivot
    quicksort_serial(arr, pivot + 1, right);
  }
}

int main(int argc, char *argv[])
{

    // Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./quicksort_serial name\n");
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
    quicksort_serial(input, 0, n-1);

    // Printing the output
    print_ans(input, n);

    return 0;
}