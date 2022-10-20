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

// #define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess) {
fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#else
#define cudaCheckError(ans) ans
#endif

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
upsweepPhaseKernel(int N,int twod1, int twod, int* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;
    
    // Check if we're inside array of size N (not nextpow(2N)) any other operation is a waste of time 
    if (index + twod1 - 1 < N - 1) {
        result[index + twod1 - 1] = result[index + twod - 1] + result[index + twod1 - 1];
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

    int index = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;

    // Check if it's worth to right swap
    // It's always worth to left swap if we right swap
    if (index + twod - 1 < N) {
        int aux = result[index + twod1 - 1];
        result[index + twod1 - 1] = result[index + twod - 1] + aux;
        result[index + twod - 1] = aux;
    }
    else {
        float counter = 0;
        //check if it's worth it to copy to the left be checking if all swaps end up inside the actual array of size N
        for (float i = twod; i >= 1; i /= 2) {
            counter += i;
        }
        if (index + twod1 - 1 - (int)counter < N) {
            result[index + twod - 1] = result[index + twod1 - 1];
        }
    }
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
initializeResultKernel(int* input, int* result, int N) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        result[index] = input[index];
    }
}

__global__ void
putZeroInEnd(int* result, int N) {
    result[N - 1] = 0;
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
    
    int nextPow2var = nextPow2(N);

    //initializeResultKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, result, N);
    //cudaCheckError(cudaDeviceSynchronize());

    // upsweep phase
    for (int twod = 1; twod < nextPow2var / 2; twod *= 2) {
        int twod1 = twod*2;
        int num_block_iter = ((N/twod1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int threads_per_block = THREADS_PER_BLOCK;
        if (num_block_iter == 1) {
            threads_per_block = (N/twod1);
        }
        upsweepPhaseKernel<<<num_block_iter, threads_per_block>>>(N, twod1, twod, result);
        cudaCheckError(cudaDeviceSynchronize());

    }

    putZeroInEnd<<<1, 1>>>(result, nextPow2var);
    cudaCheckError(cudaDeviceSynchronize());

    // downsweep phase
    for (int twod = nextPow2var / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int num_block_iter = ((nextPow2var/twod1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int threads_per_block = THREADS_PER_BLOCK;
        if (num_block_iter == 1) {
            threads_per_block = (nextPow2var/twod1);
        }
        downsweepPhaseKernel<<<num_block_iter, threads_per_block>>>(twod1, twod, result, N);
        cudaCheckError(cudaDeviceSynchronize());
    }
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
isEqualToNext(int N, int* aux, int* input) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int support[THREADS_PER_BLOCK + 1];
    
    //get all relevant information loaded in a shared array
    support[threadIdx.x] = input[index];
    if (threadIdx.x < 1) {
        support[THREADS_PER_BLOCK + threadIdx.x] = input[index + THREADS_PER_BLOCK];
    }

    __syncthreads();

    //check if element in this index has the same value of the element ahed of it and place in the new array '1' or '0'
    if (index < N - 1) {
        if (support[threadIdx.x] == support[threadIdx.x + 1]) {
            aux[index] = 1;
        }
        else {
            aux[index] = 0;
        }
    }
    
}

__global__ void
getFindRepeats(int N, int* resultarray, int* device_output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int auxiliar = 0;

    //get all relevant information loaded in a shared array
    __shared__ int support[THREADS_PER_BLOCK + 1];
    
    support[threadIdx.x] = resultarray[index];
    if (threadIdx.x < 1) {
        support[THREADS_PER_BLOCK + threadIdx.x] = resultarray[index + THREADS_PER_BLOCK];
    }
    __syncthreads();

    //check if element has a different value has the one ahead of it , if so save to the output the index. 
    //The index in the output where it will be saved is the value of the initial element
    if (index < N - 1) {
        auxiliar = support[threadIdx.x];
        if (auxiliar != support[threadIdx.x + 1]) {
            device_output[auxiliar] = index;
        }
    }

}

__global__ void
switchlast_first(int length, int* device_input) {    
    device_input[0] = device_input[length - 1];
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

    const int blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int resultarray;

    isEqualToNext<<<blocks, THREADS_PER_BLOCK>>>(length, device_output, device_input);
    cudaCheckError(cudaDeviceSynchronize());

    exclusive_scan(device_output, length, device_input);

    getFindRepeats<<<blocks, THREADS_PER_BLOCK>>>(length, device_input, device_output);
    cudaCheckError(cudaDeviceSynchronize());

    switchlast_first<<<1, 1>>>(length, device_input);
    cudaCheckError(cudaDeviceSynchronize());
    cudaMemcpy(&resultarray, device_input, sizeof(int), cudaMemcpyDeviceToHost);
    return resultarray; 
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