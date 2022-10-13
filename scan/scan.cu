#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
upsweepPhaseKernel(int twod1, int twod, int* result, int N) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if (index < N/twod1){
        index *= twod1;
        if (index + twod1 - 1 < N) {
            result[index + twod1 - 1] = result[index + twod - 1] + result[index + twod1 - 1];
        }
    }

}

// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
downsweepPhaseKernel(int twod1, int twod, int* result, int N) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if (index < N/twod1){
        index *= twod1;
        if (index + twod - 1 < N) {
            int tmp = result[index + twod - 1];
            result[index + twod - 1] = result[index + twod1 - 1];
            result[index + twod1 - 1] = tmp + result[index + twod1 - 1];
        }
    }
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
initializeResultKernel(int* input, int* result, int N, int nextPow2N) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        result[index] = input[index];
    }
    else if (index < nextPow2N) {
        result[index] = 0;
    }
}

__global__ void
putZeroInEnd(int* result, int N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index == N - 1) {
        result[index] = 0;
    }
}


// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // STUDENTS TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    
    const int blocks = (nextPow2(N) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    initializeResultKernel<<<blocks, THREADS_PER_BLOCK>>>(input, result, N, nextPow2(N));

    // Testing
    /* int* resultt = (int*)malloc(N*sizeof(int));
    cudaMemcpy(resultt, result, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Initially\n");
    for (int i = 0; i < nextPow2(N); i++) {
        printf("%d\n", resultt[i]);
    }
    printf("\n"); */

    // upsweep phase
    for (int twod = 1; twod < nextPow2(N) / 2; twod *= 2) {
        int twod1 = twod*2;
        int num_block_iter = ((nextPow2(N)/twod1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int threads_per_block = THREADS_PER_BLOCK;
        if (num_block_iter == 1) {
            threads_per_block = (nextPow2(N)/twod1);
        }
        upsweepPhaseKernel<<<num_block_iter, threads_per_block>>>(twod1, twod, result, nextPow2(N));


        // Testing
        /* cudaMemcpy(resultt, result, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Iteration %d \n", twod);
        for (int i = 0; i < nextPow2(N); i++) {
            printf("A[%d]=%d\n", i, resultt[i]);
        }
        printf("\n"); */
    }

    

    putZeroInEnd<<<blocks, THREADS_PER_BLOCK>>>(result, nextPow2(N));

     // Testing
    /* cudaMemcpy(resultt, result, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("End\n");
    for (int i = 0; i < nextPow2(N); i++) {
        printf("A[%d]=%d\n", i, resultt[i]);
    }
    printf("\n");*/

    // downsweep phase
    for (int twod = nextPow2(N) / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int num_block_iter = ((nextPow2(N)/twod1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int threads_per_block = THREADS_PER_BLOCK;
        if (num_block_iter == 1) {
            threads_per_block = (nextPow2(N)/twod1);
        }
        downsweepPhaseKernel<<<num_block_iter, threads_per_block>>>(twod1, twod, result, nextPow2(N));
    }

    // Testing
    /* cudaMemcpy(resultt, result, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("End\n");
    for (int i = 0; i < nextPow2(N); i++) {
        printf("A[%d]=%d\n", i, resultt[i]);
    }
    printf("\n"); */

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_result);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void
isEqualToNext(int N, int nextPow2N, int* aux, int* input) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N - 1) {
        if (input[index] == input[index + 1]) {
            aux[index] = 1;
        }
        else {
            aux[index] = 0;
        }
    }
    else if (index < nextPow2N) {
        aux[index] = 0;
    }
}

__global__ void
getFindRepeats(int N, int nextPow2N, int* resultarray, int* aux, int* device_output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N - 1) {
        if (aux[index] == 1) {
            device_output[resultarray[index] ] = index;
        }
    }

}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // STUDENTS TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int nextPow2var = nextPow2(length);
    // Testing
    /* int* resultt = (int*)malloc(nextPow2var*sizeof(int));
    cudaMemcpy(resultt, device_input, nextPow2var * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Initialy\n");
    for (int i = 0; i < nextPow2var; i++) {
        printf("A[%d]=%d\n", i, resultt[i]);
    }
    printf("\n"); */

    int *aux, *a, *b, *resultarray, *device_resultarray, number_pairs;
    a = (int *) malloc(nextPow2var * sizeof(int));
    if (a == NULL) {
        return -1;
    }

    cudaMalloc((void **)&aux, nextPow2var * sizeof(int));
    cudaMemcpy(aux, a, nextPow2var * sizeof(int), cudaMemcpyHostToDevice);

    const int blocks = (nextPow2var + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    isEqualToNext<<<blocks, THREADS_PER_BLOCK>>>(length, nextPow2var, aux, device_input);
    
    // Testing
    /* cudaMemcpy(resultt, aux, nextPow2var * sizeof(int), cudaMemcpyDeviceToHost);
    printf("IsEqualToNext\n");
    for (int i = 0; i < nextPow2var; i++) {
        printf("A[%d]=%d\n", i, resultt[i]);
    }
    printf("\n"); */ 

    resultarray = (int *) malloc(nextPow2var * sizeof(int));
    if (resultarray == NULL) {
        cudaFree(aux);
        free(a);
        return -1;
    }

    cudaScan(aux, aux + nextPow2var, resultarray);

    b = (int *) malloc(resultarray[nextPow2var - 1] * sizeof(int));
    if (b == NULL) {
        cudaFree(aux);
        free(a);
        free(resultarray);
        return -1;
    }
    /* for (int i = 0; i < nextPow2var; i++){
        printf("Ressultarray: %d\n", resultarray[i]);
    }
    printf("\n"); */

    number_pairs = resultarray[nextPow2var - 1]; 

    cudaMemcpy(device_output, b, number_pairs * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_resultarray, nextPow2var * sizeof(int));
    cudaMemcpy(device_resultarray, resultarray, nextPow2var* sizeof(int), cudaMemcpyHostToDevice);
    

    getFindRepeats<<<blocks, THREADS_PER_BLOCK>>>(length, nextPow2var, device_resultarray, aux, device_output);

     // Testing
    /* cudaMemcpy(resultt, device_output, number_pairs * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Device output\n");
    for (int i = 0; i < resultarray[nextPow2var - 1]; i++) {
        printf("A[%d]=%d\n", i, resultt[i]);
    }
    printf("\n"); */ 

    cudaFree(device_resultarray);
    cudaFree(aux);
    free(resultarray);
    free(a);
    free(b);
    return number_pairs; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
