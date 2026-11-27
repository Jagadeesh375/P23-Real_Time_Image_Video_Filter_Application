#include <iostream>
#include "application.h"

Application::Application()
{
    imgWidth = 0;
    imgHeight = 0;
    selectedFilter = 0;
    initialized = false;
}

Application::~Application()
{
    cleanup();
}

bool Application::initialize(const char* filterName)
{
    // Select the filter
    if (strcmp(filterName, "invert") == 0)
    {
        selectedFilter = 1;
    }
    else if (strcmp(filterName, "grayscale") == 0)
    {
        selectedFilter = 2;
    }
    else if (strcmp(filterName, "blur") == 0)
    {
        selectedFilter = 3;
    }
    else
    {
        std::cerr << "Invalid filter selection!" << std::endl;
        return false;
    }

    // Initialize camera
    if (!camera.initialize())
    {
        std::cerr << "Camera initialization failed!" << std::endl;
        return false;
    }

    // Capture one frame to determine image dimensions
    if (!camera.capture(frame))
    {
        std::cerr << "Failed to capture initial camera frame!" << std::endl;
        return false;
    }

    imgWidth = frame.cols;
    imgHeight = frame.rows;

    std::cout << "Image Width  : " << imgWidth << std::endl;
    std::cout << "Image Height : " << imgHeight << std::endl;

    // Initialize OpenGL renderer
    // Filter selection determines PBO and texture channel count
    if (!renderer.initialize(imgWidth, imgHeight, selectedFilter))
    {
        std::cerr << "OpenGL renderer initialization failed!" << std::endl;
        return false;
    }

    // Initialize the selected CUDA filter
    bool filterInitialized = false;

    if (selectedFilter == 1)
    {
        filterInitialized = inversion.initialize(imgWidth, imgHeight);
    }
    else if (selectedFilter == 2)
    {
        filterInitialized = grayscale.initialize(imgWidth, imgHeight);
    }
    else if (selectedFilter == 3)
    {
        filterInitialized = boxBlur.initialize(imgWidth, imgHeight);
    }

    if (!filterInitialized)
    {
        std::cerr << "CUDA filter initialization failed!" << std::endl;
        return false;
    }

    initialized = true;

    std::cout << "Application initialized successfully!" << std::endl;

    return true;
}

void Application::run()
{
    if (!initialized)
    {
        std::cerr << "Application is not initialized!" << std::endl;
        return;
    }

    while (!renderer.windowShouldClose())
    {
        // Capture camera frame
        if (!camera.capture(frame))
        {
            std::cerr << "Camera frame capture failed!" << std::endl;
            break;
        }

        // Map OpenGL PBO for CUDA access
        if (!renderer.mapPBO())
        {
            std::cerr << "Failed to map PBO!" << std::endl;
            break;
        }

        unsigned char* pboOutput = nullptr;
        size_t pboSize = 0;

        // Get the CUDA device pointer to the mapped PBO
        if (!renderer.getMappedPointer(&pboOutput, &pboSize))
        {
            std::cerr << "Failed to get mapped PBO pointer!" << std::endl;
            renderer.unmapPBO();
            break;
        }

        // Execute the selected CUDA filter
        bool processSuccess = false;

        if (selectedFilter == 1)
        {
            processSuccess = inversion.process(
                frame.data,
                pboOutput,
                imgWidth,
                imgHeight
            );
        }
        else if (selectedFilter == 2)
        {
            processSuccess = grayscale.process(
                frame.data,
                pboOutput,
                imgWidth,
                imgHeight
            );
        }
        else if (selectedFilter == 3)
        {
            processSuccess = boxBlur.process(
                frame.data,
                pboOutput,
                imgWidth,
                imgHeight
            );
        }

        // Unmap PBO before OpenGL accesses it
        if (!renderer.unmapPBO())
        {
            std::cerr << "Failed to unmap PBO!" << std::endl;
            break;
        }

        if (!processSuccess)
        {
            std::cerr << "CUDA filter processing failed!" << std::endl;
            break;
        }

        // Upload PBO contents into the OpenGL texture
        if (!renderer.updateTexture())
        {
            std::cerr << "Texture update failed!" << std::endl;
            break;
        }

        // Process window events
        glfwPollEvents();

        // Rendering must be performed here if your renderer
        // provides shader and render functionality.
    }

    std::cout << "Application execution completed!" << std::endl;
}

void Application::cleanup()
{
    // Release CUDA filter resources
    inversion.cleanup();
    grayscale.cleanup();
    boxBlur.cleanup();

    // Release OpenGL and CUDA-OpenGL interop resources
    renderer.cleanup();

    // Release camera resources
    camera.cleanup();

    initialized = false;
}
