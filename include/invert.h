#ifndef INVERT_H
#define INVERT_H

bool invertInit(int width, int height);

bool invertProcess(unsigned char* input, unsigned char* output, int width, int height);

void invertClean();

#endif
