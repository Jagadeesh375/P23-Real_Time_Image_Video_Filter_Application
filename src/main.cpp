
#include <iostream>
#include <cstring>

#include "application.h"

int main(int argc, char* argv[])
{
    // Step 1: Validate command-line arguments

    if (argc != 2)
    {
        std::cout << "Usage: ./image_filter [invert|grayscale|blur]\n";
        return -1;
    }

    if (strcmp(argv[1], "invert") != 0 &&
        strcmp(argv[1], "grayscale") != 0 &&
        strcmp(argv[1], "blur") != 0)
    {
        std::cout << "Invalid operation!\n";
        std::cout << "Available operations: invert, grayscale, blur\n";
        return -1;
    }

    // Step 2: Create Application object

    Application app;

    // Step 3: Initialize application

    if (!app.initialize(argv[1]))
    {
        std::cerr << "Application initialization failed\n";
        return -1;
    }

    // Step 4: Start application

    app.run();

    // Step 5: Cleanup

    app.cleanup();

    return 0;
}

