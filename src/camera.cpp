#include <iostream>
#include <opencv2/opencv.hpp>

#include "camera.h"

using namespace std;
using namespace cv;

// Camera object
static VideoCapture cap;

// Initialize camera
bool cameraInit()
{
    cap.open(0, CAP_V4L2);

    if (!cap.isOpened())
    {
        cout << "Failed to open camera!" << endl;
        return false;
    }

    cout << "Camera opened successfully!" << endl;

    return true;
}

// Capture camera frame
bool cameraCapture(Mat &frame)
{
    if (!cap.read(frame))
    {
        cout << "Failed to capture frame!" << endl;
        return false;
    }

    if (frame.empty())
    {
        cout << "Empty frame received!" << endl;
        return false;
    }

    if (frame.type() != CV_8UC3)
    {
        cout << "Unsupported camera frame format!" << endl;
        return false;
    }

    // Ensure continuous memory layout
    if (!frame.isContinuous())
    {
        frame = frame.clone();
    }

    return true;
}

// Release camera
void cameraCleanup()
{
    if (cap.isOpened())
    {
        cap.release();
    }

    cout << "Camera released successfully!" << endl;
}
