#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include <cuda_runtime.h>

class Grayscale
{
private:
    unsigned char* d_input;

    int imgWidth;
    int imgHeight;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    Grayscale();

    ~Grayscale();


    bool initialize(int width, int height);

    bool process(unsigned char* input, unsigned char* pboOutput, int width, int height);

    void cleanup();
};

#endif
