## Data
module load gcc-11.2<br />
gcc -std=c99 -lm gen_examples.c -o gen_examples<br />
./gen_examples n<br />
I have already kept a file 100input.txt for testing

## Compiling and executing the Serial version
module load gcc-11.2<br />
gcc -std=c99 -lm quicksort_serial.c -o quicksort_serial<br />
./quicksort_serial 100input.txt<br />

## Compiling and executing the OpenMP version
module load gcc-11.2<br />
gcc -fopenmp -std=c99 -lm quicksort_openmp.c -o quicksort_openmp<br />
./quicksort_openmp 100input.txt<br />

## Compiling and executing the CUDA version
module load gcc-4.9<br />
nvcc -arch=sm_35 -rdc=true quicksort_cuda.cu -o quicksort_cuda<br />
./quicksort_cuda 100input.txt<br />

All the programs print the sorted array on the screen


