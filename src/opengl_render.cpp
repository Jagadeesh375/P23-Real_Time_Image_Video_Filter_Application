#include <iostream>
#include <cstddef>

#include "opengl_render.h"


// Constructor
OpenGLRenderer::OpenGLRenderer()
{
    window = nullptr;

    VAO = 0;
    VBO = 0;
    texture = 0;
    PBO = 0;

    cudaPBO = nullptr;

    imgWidth = 0;
    imgHeight = 0;
    channels = 0;
}


// Destructor
OpenGLRenderer::~OpenGLRenderer()
{
    cleanup();
}


// Initialize GLFW, OpenGL context, Texture and PBO
bool OpenGLRenderer::initialize(int width, int height, int selectedFilter)
{
    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid image dimensions\n";
        return false;
    }

    imgWidth = width;
    imgHeight = height;

    // Select number of channels based on filter
    if (selectedFilter == 2)
    {
        channels = 1;  // Grayscale
    }
    else
    {
        channels = 3;  // Inversion and Box Blur
    }

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

    // Create window
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

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW\n";
        cleanup();
        return false;
    }

    // Clear any startup OpenGL error from GLEW

    // Set viewport
    glViewport(0, 0, imgWidth, imgHeight);


    // --------------------------------
    // Fullscreen Quad
    // --------------------------------

    float vertices[] =
    {
        // Position       // Texture coordinates

        -1.0f, -1.0f,      0.0f, 1.0f,
         1.0f, -1.0f,      1.0f, 1.0f,
        -1.0f,  1.0f,      0.0f, 0.0f,

         1.0f, -1.0f,      1.0f, 1.0f,
         1.0f,  1.0f,      1.0f, 0.0f,
        -1.0f,  1.0f,      0.0f, 0.0f
    };

    // Generate VAO and VBO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(vertices),
        vertices,
        GL_STATIC_DRAW
    );

    // Position attribute
    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)0
    );

    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)(2 * sizeof(float))
    );

    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    // --------------------------------
    // Create OpenGL Texture
    // --------------------------------

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR
    );

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR
    );

    // Allocate texture memory based on selected filter
    if (channels == 1)
    {
        // Grayscale: Single-channel texture
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            imgWidth,
            imgHeight,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }
    else
    {
        // Inversion and Box Blur: Three-channel texture
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB8,
            imgWidth,
            imgHeight,
            0,
            GL_BGR,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }

    glBindTexture(GL_TEXTURE_2D, 0);


    // --------------------------------
    // Create Pixel Buffer Object (PBO)
    // --------------------------------

    glGenBuffers(1, &PBO);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);

    // Calculate PBO memory size based on channel count
    size_t pboSize =
        static_cast<size_t>(imgWidth) * imgHeight * channels;

    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        pboSize,
        nullptr,
        GL_STREAM_DRAW
    );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


    // --------------------------------
    // Register PBO with CUDA
    // --------------------------------

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
    std::cout << "Texture created successfully\n";
    std::cout << "PBO created successfully\n";
    std::cout << "CUDA-OpenGL PBO registered successfully\n";
    std::cout << "Number of channels: " << channels << std::endl;

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


// Update OpenGL texture using PBO
bool OpenGLRenderer::updateTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);

    glBindTexture(GL_TEXTURE_2D, texture);

    // Ensure correct pixel row alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (channels == 1)
    {
        // Grayscale: One byte per pixel
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            imgWidth,
            imgHeight,
            GL_RED,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }
    else
    {
        // Color: Three bytes per pixel
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            imgWidth,
            imgHeight,
            GL_BGR,
            GL_UNSIGNED_BYTE,
            nullptr
        );
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, 0);

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

        // Delete texture
        if (texture != 0)
        {
            glDeleteTextures(1, &texture);
            texture = 0;
        }

        // Delete VBO
        if (VBO != 0)
        {
            glDeleteBuffers(1, &VBO);
            VBO = 0;
        }

        // Delete VAO
        if (VAO != 0)
        {
            glDeleteVertexArrays(1, &VAO);
            VAO = 0;
        }

        // Destroy GLFW window
        glfwDestroyWindow(window);
        window = nullptr;

        glfwTerminate();
    }

    imgWidth = 0;
    imgHeight = 0;
    channels = 0;
}
