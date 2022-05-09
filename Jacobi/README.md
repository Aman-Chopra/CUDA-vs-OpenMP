## Data
module load gcc-11.2<br />
chmod 777 ./gengs<br />
./gengs A e where A is the number of variables and e is the error<br />
I have already kept a file 10.txt for testing

## Compiling and executing the Serial version
module load gcc-11.2<br />
gcc -std=c99 -lm jacobi_serial.c -o jacobi_serial<br />
./jacobi_serial 10.txt<br />

## Compiling and executing the OpenMP version
module load gcc-11.2<br />
gcc -fopenmp -std=c99 -lm jacobi_openmp.c -o jacobi_openmp<br />
./jacobi_openmp 10.txt<br />

## Compiling and executing the CUDA version
module load gcc-4.9<br />
nvcc jacobi_cuda.cu -o jacobi_cuda<br />
./jacobi_cuda 10.txt<br />

All the programs print the coefficients and the number of iterations on the screen


