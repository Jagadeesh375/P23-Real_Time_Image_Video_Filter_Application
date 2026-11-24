#ifndef BOX_BLUR_H
#define BOX_BLUR_H

bool blurInit(int width, int height);

bool blurProcess(unsigned char* input, unsigned char* output, int width, int height);

void blurClean();

#endif
