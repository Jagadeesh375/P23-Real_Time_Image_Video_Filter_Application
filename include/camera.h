#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

class Camera
{
private:
    cv::VideoCapture cap;

public:
    Camera();

    ~Camera();

    // Prevent copying of camera resources
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool initialize();

    bool capture(cv::Mat& frame);

    void cleanup();
};

#endif

