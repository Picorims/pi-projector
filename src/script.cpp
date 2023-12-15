#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

bool verbose = false;

void debug_px(short px) {
    std::cout << std::bitset<16>(px); //bits
    //color if terminal supports it
    std::string reset = "\033[0m";
    int r = ((px & 0b0000111100000000) >> 8) * 16;
    int g = ((px & 0b0000000011110000) >> 4) * 16;
    int b = ((px & 0b0000000000001111)) * 16;
    std::cout << "(" << r << ";" << g << ";" << b << ")";
    std::cout << " \033[48;2;" << r << ";" << g << ";" << b << "m        " << reset;
    std::cout << std::endl;
}

void send_px(short px) {
    if (verbose) debug_px(px);
    // Code to send the pixel to the microcontroller goes here.
}

void processFrameSegment(const cv::Mat& frame, int startRow, int endRow, int nbFrames) {
    for(int i = startRow; i < endRow; i++) {
        for(int j = 0; j < frame.cols; j++) {
            cv::Vec3b& color = frame.at<cv::Vec3b>(i, j);
            int r16 = color[2] / 16;
            int g16 = color[1] / 16;
            int b16 = color[0] / 16;
            short pixel = (r16 << 8) + (g16 << 4) + b16;
            send_px(pixel);
            if (verbose) {
                if (!(r16 == 0 && g16 == 0 && b16 == 0)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                std::cout << "(" << i << ";" << j << ") frame " << nbFrames << ": ";
            }
        }
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }
    if (argc == 3 && argv[2] == std::string("-v")) verbose = true;

    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    auto start = std::chrono::system_clock::now();
    int nbFrames = 0;

    while(true) {
        cv::Mat frame;
        cap >> frame;
        nbFrames++;

        if(frame.empty()) {
            break;
        }

        const int numThreads = 4; // Adjust based on your Raspberry Pi's capabilities
        std::vector<std::future<void>> futures;
        int rowsPerThread = frame.rows / numThreads;

        for(int t = 0; t < numThreads; ++t) {
            int startRow = t * rowsPerThread;
            int endRow = (t + 1 == numThreads) ? frame.rows : startRow + rowsPerThread;

            futures.push_back(std::async(std::launch::async, processFrameSegment, std::cref(frame), startRow, endRow, nbFrames));
        }

        for(auto& f : futures) {
            f.get();
        }

        if(cv::waitKey(30) >= 0) {
            break;
        }
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Time (s): " << duration.count() << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
