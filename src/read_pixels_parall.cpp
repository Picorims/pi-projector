#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

bool verbose = false;

void debug_px(short px) {
    std::cout << std::bitset<16>(px);
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
}

// Fonction pour traiter une section de la frame
void process_frame_section(cv::Mat frame, int startRow, int endRow, int nbFrames) {
    for(int i = startRow; i < endRow; ++i) {
        for(int j = 0; j < frame.cols; ++j) {
            cv::Vec3b& color = frame.at<cv::Vec3b>(i, j);
            int r16 = color[2] / 16;
            int g16 = color[1] / 16;
            int b16 = color[0] / 16;
            short pixel = (r16 << 8) + (g16 << 4) + b16;
            send_px(pixel);
        }
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }
    if (argc == 3 && std::string(argv[2]) == "-v") verbose = true;

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
        if(frame.empty()) break;
        nbFrames++;

        int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        int rowsPerThread = frame.rows / numThreads;

        for(int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i + 1) * rowsPerThread;
            if(i == numThreads - 1) endRow = frame.rows;

            threads.emplace_back(process_frame_section, frame.clone(), startRow, endRow, nbFrames);
        }

        for(auto& thread : threads) {
            thread.join();
        }

        if(cv::waitKey(30) >= 0) break;
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop - start;
    auto elapsed_time = duration.count();

    std::cout << "Time (s): " << elapsed_time << std::endl;
    double timeMsPerFrame = (elapsed_time / static_cast<double>(nbFrames)) * 1000;
    std::cout << "Per frame (ms): " << timeMsPerFrame << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
