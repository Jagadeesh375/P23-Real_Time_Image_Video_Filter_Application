#ifndef BOX_BLUR_H
#define BOX_BLUR_H

#include <cuda_runtime.h>

class BoxBlur
{
private:
    unsigned char* d_input;

    int imgWidth;
    int imgHeight;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    BoxBlur();

    ~BoxBlur();

    BoxBlur(const BoxBlur&) = delete;
    BoxBlur& operator=(const BoxBlur&) = delete;

    bool initialize(int width, int height);

    bool process(unsigned char* input, unsigned char* pboOutput, int width, int height);

    void cleanup();
};

#endif
