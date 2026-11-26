#ifndef INVERSION_H
#define INVERSION_H

#include <cuda_runtime.h>

class Inversion
{
private:
    unsigned char* d_input;

    int imgWidth;
    int imgHeight;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

public:
    Inversion();

    ~Inversion();


    bool initialize(int width, int height);

    bool process(unsigned char* input, unsigned char* pboOutput, int width, int height);

    void cleanup();
};

#endif
