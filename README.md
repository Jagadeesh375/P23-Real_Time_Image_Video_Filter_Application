# Real-Time Image & Video Filter Application

A real-time image and video processing application developed using C++, OpenCV, and NVIDIA CUDA. The application captures video from a camera and performs GPU-accelerated image processing operations.

## Technologies Used

- C++
- OpenCV
- NVIDIA CUDA
- NVIDIA GPU / Jetson Platform

## Features

- Real-time video capture using OpenCV.
- GPU-accelerated grayscale conversion using CUDA.
- GPU-accelerated image blurring using CUDA.
- GPU-accelerated color inversion using CUDA.
- Common camera interface for multiple image-processing operations.

## Project Structure

P23-Real_Time_Image_Video_Filter_Application/
│
├── camera.cpp
├── blur.cu
├── blur.h
├── grayscale.cu
├── grayscale.h
├── invert.cu
├── invert.h
├── README.md
└── .gitignore


## Module Description

### 1. Camera Acquisition (`camera.cpp`)
- Captures real-time video from the connected camera using OpenCV.
- Continuously reads frames from the camera.
- Provides input frames for CUDA-based image processing.

### 2. Grayscale Processing (`grayscale.cu`)
- Converts color frames into grayscale.
- Uses CUDA kernels for parallel pixel processing.

### 3. Image Blurring (`blur.cu`)
- Applies a blur filter to the input video frames.
- Uses CUDA parallel processing for pixel-level operations.

### 4. Color Inversion (`invert.cu`)
- Performs color inversion on the input video frames.
- Uses CUDA kernels for parallel image processing.

## Header Files

The following header files contain function declarations for their respective CUDA processing modules:

- blur.h
- grayscale.h
- invert.h

## Application Workflow
        USB Camera
            |
            v
    OpenCV VideoCapture
            |
            v
        camera.cpp
            |
            v
      Input Video Frame
            |
            v
      CUDA Processing
            |
     +------+------+ 
     |      |      |
     v      v      v
 Grayscale Blur  Invert
     |      |      |
     +------+------+ 
            |
            v
     Processed Frame
            |
            v
       Display Output
