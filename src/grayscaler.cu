#include <cuda_runtime.h>
#include <iostream>

#include "grayscale.h"

using namespace std;


// GPU memory pointers

static unsigned char *d_input = nullptr;
static unsigned char *d_output = nullptr;

static int imgWidth = 0;
static int imgHeight = 0;

static cudaEvent_t startEvent = nullptr;
static cudaEvent_t stopEvent = nullptr;
// Grayscale CUDA Kernel

__global__ void grayscaleKernel(
    unsigned char *input,
    unsigned char *output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        int index = (y * width + x) * 3;

        unsigned char b = input[index];
        unsigned char g = input[index + 1];
        unsigned char r = input[index + 2];

        unsigned char gray = 0.114 * b + 0.587 * g + 0.299 * r;

        output[index]     = gray;
        output[index + 1] = gray;
        output[index + 2] = gray;
    }
}


// GPU Initialization

bool grayscaleInit(int width, int height)
{
    imgWidth = width;
    imgHeight = height;

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height * 3;

    cudaError_t err;

    // Allocate input GPU memory

    err = cudaMalloc((void**)&d_input, inputSize);

    if(err != cudaSuccess)
    {
        cerr << "Input allocation failed: " << cudaGetErrorString(err) << endl;

        return false;
    }

    // Allocate output GPU memory

    err = cudaMalloc((void**)&d_output, outputSize);

    if(err != cudaSuccess)
    {
        cerr << "Output allocation failed: " << cudaGetErrorString(err) << endl;

        cudaFree(d_input);
        d_input = nullptr;

        return false;
    }

    // Create CUDA events

    err = cudaEventCreate(&startEvent);

    if(err != cudaSuccess)
    {
        cerr << "Start event creation failed: " << cudaGetErrorString(err) << endl;

        grayscaleClean();
        return false;
    }

    err = cudaEventCreate(&stopEvent);

    if(err != cudaSuccess)
    {
        cerr << "Stop event creation failed: " << cudaGetErrorString(err) << endl;

        grayscaleClean();
        return false;
    }

    cout << "Grayscale GPU memory allocated successfully!" << endl;

    return true;
}


// Process Camera Frame

bool grayscaleProcess(unsigned char* input, unsigned char* output, int width, int height)
{
    if(!input || !output || width != imgWidth || height != imgHeight)
    {
        return false;
    }

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height * 3;

    cudaError_t err;

    // Copy input frame from CPU to GPU

    err = cudaMemcpy(d_input,input, inputSize, cudaMemcpyHostToDevice);

    if(err != cudaSuccess)
    {
        cerr << "Input copy failed: " << cudaGetErrorString(err) << endl;

        return false;
    }

    // Define CUDA block and grid

    dim3 block(16, 16);

    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    // Record starting event

    err = cudaEventRecord(startEvent);

    if(err != cudaSuccess)
        return false;

    // Execute CUDA kernel

    grayscaleKernel<<<grid, block>>>(d_input,d_output,width,height);

    // Check kernel launch error

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << endl;

        return false;
    }

    // Record stopping event

    err = cudaEventRecord(stopEvent);

    if(err != cudaSuccess)
        return false;

    // Wait for kernel execution to complete

    err = cudaEventSynchronize(stopEvent);

    if(err != cudaSuccess)
    {
        cerr << "Kernel execution failed: " << cudaGetErrorString(err) << endl;

        return false;
    }

    // Calculate kernel execution time internally

    float kernelTime = 0.0f;

    err = cudaEventElapsedTime(&kernelTime,startEvent,stopEvent);

    if(err != cudaSuccess)
        return false;
    cout << "Kernel Execution Time: " << kernelTime << " ms" << endl;

    // Copy processed frame from GPU to CPU

    err = cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);

    if(err != cudaSuccess)
    {
        cerr << "Output copy failed: " << cudaGetErrorString(err) << endl;

        return false;
    }

    return true;
}


// GPU Cleanup

void grayscaleClean()
{
    if(d_input)
        cudaFree(d_input);

    if(d_output)
        cudaFree(d_output);

    if(startEvent)
        cudaEventDestroy(startEvent);

    if(stopEvent)
        cudaEventDestroy(stopEvent);

    d_input = nullptr;
    d_output = nullptr;

    startEvent = nullptr;
    stopEvent = nullptr;

    imgWidth = 0;
    imgHeight = 0;

    cout << "Grayscale GPU resources released!" << endl;
}
