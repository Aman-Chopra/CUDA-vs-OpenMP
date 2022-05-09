## Data
n = the size of the input vectors

## Compiling and executing the Serial version
module load gcc-11.2<br />
gcc -std=c99 -lm vector_add_serial.c -o vector_add_serial<br />
./vector_add_serial 10<br />

## Compiling and executing the OpenMP version
module load gcc-11.2<br />
gcc -fopenmp -std=c99 -lm vector_add_openmp.c -o vector_add_openmp<br />
./vector_add_openmp 10<br />

## Compiling and executing the CUDA version
module load gcc-4.9<br />
nvcc vector_add_cuda.cu -o vector_add_cuda<br />
./vector_add_openmp 10<br />

All the programs print the input and sum vectors on the screen



