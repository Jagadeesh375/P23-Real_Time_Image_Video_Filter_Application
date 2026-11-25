#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "camera.h"
#include "invert.h"
#include "grayscale.h"
#include "box_blur.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // 1. Validate command-line arguments

    if (argc != 2)
    {
        cout << "Usage: ./image_filter [invert|grayscale|blur]" << endl;
        return -1;
    }

    if (strcmp(argv[1], "invert") != 0 &&
        strcmp(argv[1], "grayscale") != 0 &&
        strcmp(argv[1], "blur") != 0)
    {
        cout << "Invalid operation!" << endl;
        cout << "Available operations: invert, grayscale, blur" << endl;
        return -1;
    }

    // 2. Initialize camera

    if (!cameraInit())
    {
        cout << "Camera initialization failed!" << endl;
        return -1;
    }

    Mat frame;
    Mat output;

    bool initialized = false;

    // 3. Main application loop

    while (true)
    {
        // Capture camera frame

        if (!cameraCapture(frame))
        {
            cout << "Camera frame capture failed!" << endl;
            break;
        }

        // Allocate output frame

        output.create(frame.rows, frame.cols, CV_8UC3);

        // 4. Initialize selected CUDA module

        if (!initialized)
        {
            bool initSuccess = false;

            if (strcmp(argv[1], "invert") == 0)
            {
                initSuccess = invertInit(frame.cols, frame.rows);
            }
            else if (strcmp(argv[1], "grayscale") == 0)
            {
                initSuccess = grayscaleInit(frame.cols, frame.rows);
            }
            else if (strcmp(argv[1], "blur") == 0)
            {
                initSuccess = blurInit(frame.cols, frame.rows);
            }

            if (!initSuccess)
            {
                cout << "CUDA initialization failed!" << endl;
                break;
            }

            initialized = true;
        }

        // 5. Process frame using selected CUDA module

        bool processSuccess = false;

        if (strcmp(argv[1], "invert") == 0)
        {
            processSuccess = invertProcess(frame.data, output.data, frame.cols, frame.rows);
        }
        else if (strcmp(argv[1], "grayscale") == 0)
        {
            processSuccess = grayscaleProcess(frame.data, output.data, frame.cols, frame.rows);
        }
        else if (strcmp(argv[1], "blur") == 0)
        {
            processSuccess = blurProcess(frame.data, output.data, frame.cols, frame.rows);
        }

        if (!processSuccess)
        {
            cout << "CUDA processing failed!" << endl;
            break;
        }

        // 6. Display original and processed frames

        imshow("Original Frame", frame);
        imshow("Processed Frame", output);

        // Exit when user presses ESC

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    // 7. Cleanup CUDA resources

    if (initialized)
    {
        if (strcmp(argv[1], "invert") == 0)
        {
            invertClean();
        }
        else if (strcmp(argv[1], "grayscale") == 0)
        {
            grayscaleClean();
        }
        else if (strcmp(argv[1], "blur") == 0)
        {
            blurClean();
        }
    }

    // 8. Cleanup camera resources

    cameraCleanup();

    destroyAllWindows();

    return 0;
}
