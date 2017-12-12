#define WINDOWS 1

#ifdef WINDOWS
// Import these libraries if using MS Visual Studio for development.
// They are needed by nvcc to interface with MS Visual Studio.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif // !WINDOWS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024       // DEFAULT values for a, b and c
#define DATATYPE int    // datatype used. For getting GFLOPS use float.
#define MAX_RANDOM 20   // Random limit
#define BLOCK_SIZE 32   // DEFAULT block size

/**
 * Kernel for multiplication of matrices
 * Each thread calculates the element of the resultant matrix
 * The thread decides the element using block_index and thread_index values
 * Since it calculates the resultant with help of grid, I did not need a call to __syncThreads()
 */
__global__ 
void multiply(DATATYPE *p, DATATYPE *q, DATATYPE *r, int a, int b, int c) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;    // get the row index for resultant matrix
    int col = blockDim.x * blockIdx.x + threadIdx.x;    // get the column index for column matrix
    int sum = 0, i = 0;
    if (row < a && col < c) {
        for (i = 0; i < b; ++i) {
            sum += p[row * b + i] * q[i * c + col];
        }
        r[row * c + col] = sum;
    }
    
}

int main(int argc, char *argv[]) {
    // declarations
    int a = SIZE, b = SIZE, c = SIZE;
    DATATYPE *p, *q, *r, *hr;
    int i, j, k;
    float time_elapsed = 0;

    DATATYPE *d_p, *d_q, *d_r;
    int N_BLOCKS = BLOCK_SIZE, N_THREADS;

    cudaEvent_t start, stop;

    // read from command line if possible
    if (argc == 4) {
        a = atoi(argv[1]);
        b = atoi(argv[2]);
        c = atoi(argv[3]);
        printf("Reading values of a, b and c\na = %d, b = %d, c = %d\n", a, b, c);
    } else { // otherwise proceed with default values
        fprintf(stderr, "proceeding with default values of a, b and c (1024)\n");
        fprintf(stderr, "usage: kernel a b c\n");
    }
    
    // get number of blocks and threads to use from user
    printf("Enter N_BLOCKS: ");
    scanf("%d", &N_BLOCKS);
    printf("Enter N_THREADS: ");
    scanf("%d", &N_THREADS);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory using unified memory model
    cudaMallocHost((void**) &p, sizeof(DATATYPE) * a * b);
    cudaMallocHost((void**) &q, sizeof(DATATYPE) * b * c);
    cudaMallocHost((void**) &r, sizeof(DATATYPE) * a * c);

    // create random variable using seed and initialize matrices
    // Initialize matrix c to 0
    srand(time(NULL));
    for (i = 0; i < a; ++i) {
        for (j = 0; j < b; ++j) {
            p[i * b + j] = rand() % MAX_RANDOM;
        }
    }
    for (i = 0; i < b; ++i) {
        for (j = 0; j < c; ++j) {
            q[i * c + j] = rand() % MAX_RANDOM;
        }
    }
    for (i = 0; i < a; ++i) {
        for (j = 0; j < b; ++j) {
            r[i * c + j] = 0;
        }
    }

    // find the optimal configuration of number of threads
    // create a dim3 structure for it
    unsigned int rows = (a + N_BLOCKS - 1) / N_BLOCKS;
    unsigned int cols = (c + N_BLOCKS - 1) / N_BLOCKS;
    dim3 dimGrid(cols, rows);
    dim3 dimBlock(N_BLOCKS, N_BLOCKS);

    // start the clock
    cudaEventRecord(start);
    // allocate device memory
    cudaMalloc((void **) &d_p, sizeof(DATATYPE) * a * b);
    cudaMalloc((void **) &d_q, sizeof(DATATYPE) * b * c);
    cudaMalloc((void **) &d_r, sizeof(DATATYPE) * a * c);

    // copy the contents from host to device
    cudaMemcpy(d_p, p, sizeof(DATATYPE) * a * b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, sizeof(DATATYPE) * b * c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(DATATYPE) * a * c, cudaMemcpyHostToDevice);

    // call kernel
    multiply <<< dimGrid, dimBlock >>> (d_p, d_q, d_r, a, b, c);
    
    // copy the results back and let all threads sync
    cudaMemcpy(r, d_r, sizeof(DATATYPE) * a * c, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    //stop clock and free memory
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_r);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // find the time taken
    cudaEventElapsedTime(&time_elapsed, start, stop);
    // and print it
    printf("Time: %f msecs\n", time_elapsed);

    // free host memory
    cudaFreeHost(p);
    cudaFreeHost(q);
    cudaFreeHost(r);

    return 0;
}
