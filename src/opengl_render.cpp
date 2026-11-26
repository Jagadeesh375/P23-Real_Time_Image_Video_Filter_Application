
#include <iostream>
#include <cstddef>

#include "opengl_render.h"

// Constructor
OpenGLRenderer::OpenGLRenderer()
{
    window = nullptr;

    PBO = 0;
    cudaPBO = nullptr;

    imgWidth = 0;
    imgHeight = 0;
}

// Destructor
OpenGLRenderer::~OpenGLRenderer()
{
    cleanup();
}

// Initialize GLFW, OpenGL context and PBO
bool OpenGLRenderer::initialize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid image dimensions\n";
        return false;
    }

    imgWidth = width;
    imgHeight = height;

    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    // Request OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create GLFW window
    window = glfwCreateWindow(
        imgWidth,
        imgHeight,
        "Real-Time Image and Video Filter",
        nullptr,
        nullptr
    );

    if (window == nullptr)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    // Make OpenGL context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW\n";
        cleanup();
        return false;
    }

    // Clear GLEW startup error
    glGetError();

    // Set viewport
    glViewport(0, 0, imgWidth, imgHeight);

    // Create Pixel Buffer Object
    glGenBuffers(1, &PBO);

    // Bind PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);

    // Allocate PBO memory
    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        imgWidth * imgHeight * 3,
        nullptr,
        GL_STREAM_DRAW
    );

    // Unbind PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(
        &cudaPBO,
        PBO,
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "CUDA PBO registration failed: "
                  << cudaGetErrorString(cudaStatus)
                  << std::endl;

        cleanup();
        return false;
    }

    std::cout << "OpenGL initialized successfully\n";
    std::cout << "CUDA PBO registered successfully\n";

    return true;
}

// Map PBO for CUDA access
bool OpenGLRenderer::mapPBO()
{
    cudaError_t status = cudaGraphicsMapResources(
        1,
        &cudaPBO,
        0
    );

    if (status != cudaSuccess)
    {
        std::cerr << "PBO mapping failed: "
                  << cudaGetErrorString(status)
                  << std::endl;

        return false;
    }

    return true;
}

// Get mapped CUDA device pointer
bool OpenGLRenderer::getMappedPointer(
    unsigned char** ptr,
    size_t* size)
{
    if (ptr == nullptr || size == nullptr)
    {
        std::cerr << "Invalid pointer arguments\n";
        return false;
    }

    void* devicePointer = nullptr;

    cudaError_t status = cudaGraphicsResourceGetMappedPointer(
        &devicePointer,
        size,
        cudaPBO
    );

    if (status != cudaSuccess)
    {
        std::cerr << "Getting mapped pointer failed: "
                  << cudaGetErrorString(status)
                  << std::endl;

        return false;
    }

    *ptr = static_cast<unsigned char*>(devicePointer);

    return true;
}

// Unmap PBO after CUDA processing
bool OpenGLRenderer::unmapPBO()
{
    cudaError_t status = cudaGraphicsUnmapResources(
        1,
        &cudaPBO,
        0
    );

    if (status != cudaSuccess)
    {
        std::cerr << "PBO unmapping failed: "
                  << cudaGetErrorString(status)
                  << std::endl;

        return false;
    }

    return true;
}

// Check whether window should close
bool OpenGLRenderer::windowShouldClose()
{
    if (window == nullptr)
    {
        return true;
    }

    return glfwWindowShouldClose(window);
}

// Cleanup OpenGL and CUDA resources
void OpenGLRenderer::cleanup()
{
    // Unregister CUDA resource
    if (cudaPBO != nullptr)
    {
        cudaGraphicsUnregisterResource(cudaPBO);
        cudaPBO = nullptr;
    }

    if (window != nullptr)
    {
        glfwMakeContextCurrent(window);

        // Delete PBO
        if (PBO != 0)
        {
            glDeleteBuffers(1, &PBO);
            PBO = 0;
        }

        // Destroy GLFW window
        glfwDestroyWindow(window);
        window = nullptr;

        glfwTerminate();
    }

    imgWidth = 0;
    imgHeight = 0;
}
