## Data
n = the number till which the prime numbers need to be generated

## Compiling and executing the Serial version
module load gcc-11.2<br />
gcc -std=c99 -lm sieve_serial.c -o sieve_serial<br />
./sieve_serial 101<br />

## Compiling and executing the OpenMP version
module load gcc-11.2<br />
gcc -fopenmp -std=c99 -lm sieve_openmp.c -o sieve_openmp<br />
./sieve_openmp 101<br />

## Compiling and executing the CUDA version
module load gcc-4.9<br />
nvcc sieve_cuda.cu -o sieve_cuda<br />
./sieve_cuda 101<br />

All the programs print the output on the screen



