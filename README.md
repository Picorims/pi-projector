# pi-projector
Development of the projector preprocessing (pixels processing) for the raspberry PI. (Academic project)

## Setup

Install wiring pi by putting the `WiringPi` folder into src. When opening it, it should lead to the root of the git clone of the repository of the library.


## Compiling

g++ src/read_video.cpp -o read_video.out `pkg-config --cflags --libs opencv4`

## Bus

UARIT