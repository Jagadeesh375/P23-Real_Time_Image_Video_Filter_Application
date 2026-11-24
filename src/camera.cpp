#include<iostream>
#include<opencv2/opencv.hpp>

#include "invert.h"
#include "grayscale.h"
#include "box_blur.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // Check command-line arguments

    if(argc != 2)
    {
        cout<<"Usage: ./image_filter [invert|grayscale|blur]"<<endl;
        return -1;
    }

    string operation = argv[1];

    // Validate filter selection

    if(operation != "invert" && operation != "grayscale" && operation != "blur")
    {
        cerr<<"Invalid operation!"<<endl;
        return -1;
    }

    // Camera initialization

    VideoCapture cap(0, CAP_V4L2);

    if(!cap.isOpened())
    {
        cerr<<"Camera initialization failed!"<<endl;
        return -1;
    }

    Mat frame, output;

    bool state = false;

    cout<<"Camera opened successfully!"<<endl;

    while(true)
    {
        // Capture camera frame

        if(!cap.read(frame))
        {
            cerr<<"Frame capturing failed!"<<endl;
            break;
        }

        if(frame.empty() || frame.type() != CV_8UC3)
        {
            cerr<<"Invalid frame format!"<<endl;
            break;
        }

        // Ensure continuous memory

        if(!frame.isContinuous())
        {
            frame = frame.clone();
        }

        // GPU initialization - only once

        if(!state)
        {
            output.create(frame.rows,frame.cols,CV_8UC3);

            bool status = false;

            if(operation == "invert")
            {
        status = invertInit(frame.cols,frame.rows);
            }
            else if(operation == "grayscale")
            {
                status = grayscaleInit(frame.cols,frame.rows);
            }
            else if(operation == "blur")
            {
                status = blurInit(frame.cols, frame.rows);     
            }

            if(!status)
            {
                cerr<<"GPU memory allocation failed!"<<endl;
                break;
            }

            state = true;
        }

        // CUDA image processing

        bool processStatus = false;

        if(operation == "invert")
        {
            processStatus = invertProcess(frame.data,output.data,frame.cols,frame.rows);
        }
        else if(operation == "grayscale")
        {
            processStatus = grayscaleProcess(frame.data,output.data,frame.cols,frame.rows);
           
        }
        else if(operation == "blur")
        {
            processStatus = blurProcess(frame.data,output.data,frame.cols,frame.rows);
        }

        if(!processStatus)
        {
            cerr<<"CUDA processing failed!"<<endl;
            break;
        }

        // Display original and processed frames

        imshow("Original Image", frame);

        imshow("Processed Image", output);

        // Exit when ESC is pressed

        if(waitKey(1) == 27)
        {
            break;
        }
    }

    // GPU resource cleanup

    if(state)
    {
        if(operation == "invert")
        {
            invertClean();
        }
        else if(operation == "grayscale")
        {
            grayscaleClean();
        }
        else if(operation == "blur")
        {
            blurClean();
        }
    }

    // Release camera

    cap.release();

    destroyAllWindows();

    return 0;
}
