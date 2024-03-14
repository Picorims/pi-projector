#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#define WIDTH 100
#define HEIGHT 100
#define BUFFER_SIZE 10
bool verbose = false;


// https://stackoverflow.com/questions/56048952/is-it-possible-to-implement-a-thread-safe-circular-bufffer-that-consists-of-arra

class circular_buffer_t {
private:
    short buffer[BUFFER_SIZE][WIDTH][HEIGHT] = {};
    bool readable_frames[BUFFER_SIZE] = {};
    int read_pos = 0;
    int write_pos = BUFFER_SIZE - 1; //next write pos is 0

public:
    circular_buffer_t() {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            readable_frames[i] = false;
        }
    }

    int next_read_pos() {
        return (read_pos+1) % BUFFER_SIZE;
    }

    int next_write_pos() {
        return (write_pos+1) % BUFFER_SIZE;
    }

    void incr_read_pos() {
        readable_frames[read_pos] = false;
        read_pos = next_read_pos();
    }

    void incr_write_pos() {
        readable_frames[write_pos] = true;
        write_pos = next_write_pos();
    }

    bool next_write_pos_available() {
        return readable_frames[next_write_pos()] == false;
    }

    bool current_frame_readable() {
        return readable_frames[read_pos];
    }

    short read(int i, int j) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);

        return buffer[read_pos][i][j];
    }

    void write(int i, int j, short v) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);

        buffer[write_pos][i][j] = v;
    }
};


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
void process_frame_section(cv::Mat frame, int startRow, int endRow, int nbFrames, std::shared_ptr<circular_buffer_t> buf) {
    for(int i = startRow; i < endRow; ++i) {
        for(int j = 0; j < frame.cols; ++j) {
            cv::Vec3b& color = frame.at<cv::Vec3b>(i, j);
            int r16 = color[2] / 16;
            int g16 = color[1] / 16;
            int b16 = color[0] / 16;
            short pixel = (r16 << 8) + (g16 << 4) + b16;
            buf->write(i,j,pixel);
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
    std::shared_ptr<circular_buffer_t> frame_buffer = std::make_shared<circular_buffer_t>();

    while(true) {
        if (frame_buffer->next_write_pos_available()) {
            frame_buffer->incr_write_pos();
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

                threads.emplace_back(process_frame_section, frame.clone(), startRow, endRow, nbFrames, frame_buffer);
            }

            for(auto& thread : threads) {
                thread.join();
            }

            if(cv::waitKey(30) >= 0) break;
        } else {
            if (frame_buffer->current_frame_readable()) {    
                for (int i = 0; i < WIDTH; i++) {
                    for (int j = 0; j < HEIGHT; j++) {
                        send_px(frame_buffer->read(i,j));
                    }
                frame_buffer->incr_read_pos();
                }
            }
        }
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop - start;
    auto elapsed_time = duration.count();

    std::cout << "Time (s): " << elapsed_time << std::endl;
    double timeMsPerFrame = (elapsed_time / static_cast<double>(nbFrames)) * 1000;
    std::cout << "Per frame (ms): " << timeMsPerFrame << std::endl;
    double timeMsPerPixel = (timeMsPerFrame / static_cast<double>(WIDTH * HEIGHT));
    std::cout << "Per pixel (ms): " << timeMsPerPixel << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
