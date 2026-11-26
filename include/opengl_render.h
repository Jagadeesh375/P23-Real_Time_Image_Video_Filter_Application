
#ifndef OPENGL_RENDER_H
#define OPENGL_RENDER_H

#include <cstddef>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class OpenGLRenderer
{
private:

    // GLFW window
    GLFWwindow* window;

    // OpenGL resources
    GLuint VAO;
    GLuint VBO;

    // Pixel Buffer Object
    GLuint PBO;

    // CUDA graphics resource
    cudaGraphicsResource* cudaPBO;

    // Image dimensions
    int imgWidth;
    int imgHeight;

public:

    OpenGLRenderer();
    ~OpenGLRenderer();

    // Initialization
    bool initialize(int width, int height);

    // CUDA-PBO interoperability
    bool mapPBO();
    bool getMappedPointer(unsigned char** ptr, size_t* size);
    bool unmapPBO();

    // Texture and rendering

    // Window management
    bool windowShouldClose();

    // Resource cleanup
    void cleanup();
};

#endif
