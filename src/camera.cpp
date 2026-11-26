#include <iostream>
#include "camera.h"

Camera::Camera()
{
    std::cout << "Camera object created\n";
}

Camera::~Camera()
{
    cleanup();
}

bool Camera::initialize()
{
    // Open the default camera: /dev/video0
    if (!cap.open(0, cv::CAP_V4L2))
    {
        std::cerr << "Failed to open camera\n";
        return false;
    }

    std::cout << "Camera initialized successfully\n";

    return true;
}

bool Camera::capture(cv::Mat& frame)
{
    if (!cap.read(frame))
    {
        std::cerr << "Failed to capture frame\n";
        return false;
    }

    if (frame.empty())
    {
        std::cerr << "Captured frame is empty\n";
        return false;
    }

    // Expect 8-bit, 3-channel BGR image
    if (frame.type() != CV_8UC3)
    {
        std::cerr << "Unsupported camera frame format\n";
        return false;
    }

    // Ensure continuous memory for CUDA processing
    if (!frame.isContinuous())
    {
        frame = frame.clone();
    }

    return true;
}

void Camera::cleanup()
{
    if (cap.isOpened())
    {
        cap.release();

        std::cout << "Camera released successfully\n";
    }
}

