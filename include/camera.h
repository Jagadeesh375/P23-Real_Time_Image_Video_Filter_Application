#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

bool cameraInit();
bool cameraCapture(cv::Mat &frame);
void cameraCleanup();

#endif
