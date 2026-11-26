
#include <iostream>
#include <cstddef>
#include "invert.h"

__global__ void inversionKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        size_t index = (static_cast<size_t>(y) * width + x) * 3;

        output[index]     = 255 - input[index];
        output[index + 1] = 255 - input[index + 1];
        output[index + 2] = 255 - input[index + 2];
    }
}

Inversion::Inversion()
{
    d_input = nullptr;

    imgWidth = 0;
    imgHeight = 0;

    startEvent = nullptr;
    stopEvent = nullptr;
}

Inversion::~Inversion()
{
    cleanup();
}

bool Inversion::initialize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid image dimensions\n";
        return false;
    }

    cleanup();

    imgWidth = width;
    imgHeight = height;

    size_t size = static_cast<size_t>(width) * height * 3;

    // Allocate only input memory
    if (cudaMalloc(reinterpret_cast<void**>(&d_input), size) != cudaSuccess)
    {
        std::cerr << "Failed to allocate input memory\n";
        cleanup();
        return false;
    }

    if (cudaEventCreate(&startEvent) != cudaSuccess)
    {
        std::cerr << "Failed to create start event\n";
        cleanup();
        return false;
    }

    if (cudaEventCreate(&stopEvent) != cudaSuccess)
    {
        std::cerr << "Failed to create stop event\n";
        cleanup();
        return false;
    }

    std::cout << "Inversion initialized successfully\n";

    return true;
}

bool Inversion::process(unsigned char* input, unsigned char* pboOutput, int width, int height)
{
    if (input == nullptr || pboOutput == nullptr ||
        d_input == nullptr || width != imgWidth || height != imgHeight)
    {
        std::cerr << "Invalid input or image dimensions\n";
        return false;
    }

    size_t size = static_cast<size_t>(width) * height * 3;

    // Step 1: Copy camera frame from host to CUDA device
    if (cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Host to device copy failed\n";
        return false;
    }

    // Step 2: Configure CUDA grid and block
    dim3 block(16, 16);

    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Step 3: Start kernel timing
    if (cudaEventRecord(startEvent) != cudaSuccess)
    {
        std::cerr << "Failed to record start event\n";
        return false;
    }

    // Step 4: Launch kernel
    // Output is written directly into the mapped OpenGL PBO
    inversionKernel<<<grid, block>>>(d_input, pboOutput, width, height);

    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Inversion kernel launch failed\n";
        return false;
    }

    // Step 5: Wait for kernel completion
    if (cudaEventRecord(stopEvent) != cudaSuccess ||
        cudaEventSynchronize(stopEvent) != cudaSuccess)
    {
        std::cerr << "Kernel execution or synchronization failed\n";
        return false;
    }

    // Step 6: Calculate kernel execution time
    float elapsedTime = 0.0f;

    if (cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent) == cudaSuccess)
    {
        std::cout << "Inversion kernel time: " << elapsedTime << " ms\n";
    }

    // No device-to-host copy is required.
    // CUDA has already written the output into the mapped PBO.

    return true;
}

void Inversion::cleanup()
{
    if (d_input != nullptr)
    {
        cudaFree(d_input);
        d_input = nullptr;
    }

    if (startEvent != nullptr)
    {
        cudaEventDestroy(startEvent);
        startEvent = nullptr;
    }

    if (stopEvent != nullptr)
    {
        cudaEventDestroy(stopEvent);
        stopEvent = nullptr;
    }

    imgWidth = 0;
    imgHeight = 0;
}

