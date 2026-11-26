#include <iostream>
#include <cstring>

#include "application.h"

Application::Application()
{
    imgWidth = 0;
    imgHeight = 0;

    initialized = false;
    selectedFilter = 0;
}

Application::~Application()
{
    cleanup();
}

bool Application::initialize(const char* filterName)
{
    // Step 1: Validate filter selection

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
        std::cerr << "Invalid filter selection\n";
        return false;
    }

    // Step 2: Initialize camera

    if (!camera.initialize())
    {
        std::cerr << "Camera initialization failed\n";
        return false;
    }

    // Step 3: Capture initial frame

    if (!camera.capture(frame))
    {
        std::cerr << "Initial camera frame capture failed\n";
        cleanup();
        return false;
    }

    // Step 4: Get image dimensions

    imgWidth = frame.cols;
    imgHeight = frame.rows;

    // Step 5: Initialize OpenGL renderer

    if (!renderer.initialize(imgWidth, imgHeight))
    {
        std::cerr << "OpenGL renderer initialization failed\n";
        cleanup();
        return false;
    }

    // Step 6: Initialize selected CUDA filter

    bool initSuccess = false;

    if (selectedFilter == 1)
    {
        initSuccess = inversion.initialize(imgWidth, imgHeight);
    }
    else if (selectedFilter == 2)
    {
        initSuccess = grayscale.initialize(imgWidth, imgHeight);
    }
    else if (selectedFilter == 3)
    {
        initSuccess = boxBlur.initialize(imgWidth, imgHeight);
    }

    if (!initSuccess)
    {
        std::cerr << "CUDA initialization failed\n";
        cleanup();
        return false;
    }

    initialized = true;

    std::cout << "Application initialized successfully\n";

    return true;
}

void Application::run()
{
    while (!renderer.windowShouldClose())
    {
        // Step 1: Capture camera frame

        if (!camera.capture(frame))
        {
            std::cerr << "Camera frame capture failed\n";
            break;
        }

        // Step 2: Map OpenGL PBO

        if (!renderer.mapPBO())
        {
            std::cerr << "PBO mapping failed\n";
            break;
        }

        // Step 3: Get mapped PBO device pointer

        unsigned char* pboOutput = nullptr;
        size_t pboSize = 0;

        if (!renderer.getMappedPointer(&pboOutput, &pboSize))
        {
            std::cerr << "Failed to get PBO pointer\n";
            renderer.unmapPBO();
            break;
        }

        // Step 4: Process frame using selected CUDA filter

        bool processSuccess = false;

        if (selectedFilter == 1)
        {
            processSuccess = inversion.process(frame.data, pboOutput, imgWidth, imgHeight);
        }
        else if (selectedFilter == 2)
        {
            processSuccess = grayscale.process(frame.data, pboOutput, imgWidth, imgHeight);
        }
        else if (selectedFilter == 3)
        {
            processSuccess = boxBlur.process(frame.data, pboOutput, imgWidth, imgHeight);
        }

        // Step 5: Unmap PBO

        if (!renderer.unmapPBO())
        {
            std::cerr << "PBO unmapping failed\n";
            break;
        }

        if (!processSuccess)
        {
            std::cerr << "CUDA processing failed\n";
            break;
        }

    }
}

void Application::cleanup()
{
    if (initialized)
    {
        if (selectedFilter == 1)
        {
            inversion.cleanup();
        }
        else if (selectedFilter == 2)
        {
            grayscale.cleanup();
        }
        else if (selectedFilter == 3)
        {
            boxBlur.cleanup();
        }

        initialized = false;
    }

    renderer.cleanup();

    camera.cleanup();

    std::cout << "Application cleanup completed\n";
}
