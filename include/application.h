#ifndef APPLICATION_H
#define APPLICATION_H

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "invert.h"
#include "grayscale.h"
#include "box_blur.h"
#include "opengl_render.h"

class Application
{
private:

    // Camera module
    Camera camera;

    // CUDA filter modules
    Inversion inversion;
    Grayscale grayscale;
    BoxBlur boxBlur;

    // OpenGL renderer
    OpenGLRenderer renderer;

    // Captured camera frame
    cv::Mat frame;

    // Image dimensions
    int imgWidth;
    int imgHeight;

    int selectedFilter;

    // Application initialization status
    bool initialized;

public:

    Application();

    ~Application();

    // Initialize camera, OpenGL renderer and selected CUDA filter
    bool initialize(const char* filterName);

    // Capture, process and display camera frames
    void run();

    // Release all allocated resources
    void cleanup();
};

#endif
