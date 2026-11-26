#include <iostream>
#include <cstddef>
#include"grayscale.h"
__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3;

        unsigned char B = input[index];
        unsigned char G = input[index + 1];
        unsigned char R = input[index + 2];

        unsigned char gray = 0.114f * B + 0.587f * G + 0.299f * R;

        output[index]     = gray;
        output[index + 1] = gray;
        output[index + 2] = gray;
    }
}

Grayscale::Grayscale()
{
    d_input = nullptr;

    imgWidth = 0;
    imgHeight = 0;

    startEvent = nullptr;
    stopEvent = nullptr;
}

Grayscale::~Grayscale()
{
    cleanup();
}

bool Grayscale::initialize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid image dimensions\n";
        return false;
    }


    imgWidth = width;
    imgHeight = height;

    size_t size = width * height * 3;

    if (cudaMalloc(&d_input, size) != cudaSuccess)
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

    std::cout << "Grayscale initialized successfully\n";

    return true;
}

bool Grayscale::process(unsigned char* input, unsigned char* pboOutput, int width, int height)
{
    if (input == nullptr || pboOutput == nullptr || d_input == nullptr ||
        width != imgWidth || height != imgHeight)
    {
        std::cerr << "Invalid input or image dimensions\n";
        return false;
    }

    size_t size = width * height * 3;

    if (cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Host to device copy failed\n";
        return false;
    }

    dim3 block(16, 16);

    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    cudaEventRecord(startEvent);

    grayscaleKernel<<<grid, block>>>(d_input, pboOutput, width, height);

    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Grayscale kernel launch failed\n";
        return false;
    }

    cudaEventRecord(stopEvent);

    if (cudaEventSynchronize(stopEvent) != cudaSuccess)
    {
        std::cerr << "CUDA event synchronization failed\n";
        return false;
    }

    float elapsedTime = 0.0f;

    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Grayscale kernel time: " << elapsedTime << " ms\n";

    return true;
}

void Grayscale::cleanup()
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
