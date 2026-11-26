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

    Camera camera;

    Inversion inversion;
    Grayscale grayscale;
    BoxBlur boxBlur;

    OpenGLRenderer renderer;

    cv::Mat frame;

    int imgWidth;
    int imgHeight;

    int selectedFilter;

    bool initialized;

public:

    Application();

    ~Application();

    bool initialize(const char* filterName);

    void run();

    void cleanup();
};

#endif
