#include <iostream>
#include <cstddef>
#include "box_blur.h"

__global__ void boxBlurKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int sumB = 0;
        int sumG = 0;
        int sumR = 0;
        int count = 0;

        for (int dy = -3; dy <= 3; dy++)
        {
            for (int dx = -3; dx <= 3; dx++)
            {
                int sx = x + dx;
                int sy = y + dy;

                if (sx >= 0 && sx < width && sy >= 0 && sy < height)
                {
                    size_t index = (static_cast<size_t>(sy) * width + sx) * 3;

                    sumB += input[index];
                    sumG += input[index + 1];
                    sumR += input[index + 2];

                    count++;
                }
            }
        }

        size_t outputIndex = (y * width + x) * 3;

        output[outputIndex]     = sumB / count;
        output[outputIndex + 1] = sumG / count;
        output[outputIndex + 2] = sumR / count;
    }
}

BoxBlur::BoxBlur()
{
    d_input = nullptr;

    imgWidth = 0;
    imgHeight = 0;

    startEvent = nullptr;
    stopEvent = nullptr;
}

BoxBlur::~BoxBlur()
{
    cleanup();
}

bool BoxBlur::initialize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid image dimensions\n";
        return false;
    }

    cleanup();

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

    std::cout << "BoxBlur initialized successfully\n";

    return true;
}

bool BoxBlur::process(unsigned char* input, unsigned char* pboOutput, int width, int height)
{
    if (input == nullptr || pboOutput == nullptr || d_input == nullptr ||
        width != imgWidth || height != imgHeight)
    {
        std::cerr << "Invalid input or image dimensions\n";
        return false;
    }

    size_t size = static_cast<size_t>(width) * height * 3;

    if (cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cerr << "Host to device copy failed\n";
        return false;
    }

    dim3 block(16, 16);

    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEventRecord(startEvent);

    boxBlurKernel<<<grid, block>>>(d_input, pboOutput, width, height);

    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "BoxBlur kernel launch failed\n";
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

    std::cout << "BoxBlur kernel time: " << elapsedTime << " ms\n";

    return true;
}

void BoxBlur::cleanup()
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
