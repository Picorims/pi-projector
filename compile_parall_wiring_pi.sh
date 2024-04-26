g++ -DWIRING_PI src/read_pixels_parall.cpp -o read_pixels_parall.out `pkg-config --cflags --libs opencv4` -lpthread -lwiringPi
