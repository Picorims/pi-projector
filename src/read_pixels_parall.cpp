#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <unistd.h>

#ifdef WIRING_PI
#include <wiringPi.h>
#include <wiringPiSPI.h>
#endif

#define WIDTH 100
#define HEIGHT 100
#define FPS 10
#define BUFFER_SIZE 10
bool flagVerbose = false;
bool flagSpiEnabled = true;
bool flagGraphicDisp = false;
bool flagDumpBuffer = false;
bool flagVerboseBuffer = false;
bool flagLaserSim = false;

// Initialisation de WiringPi SPI
const int SPI_CHANNEL = 0; // Utilisez le canal 0 de SPI
const int SPI_SPEED = 1000000; // 1 MHz




// https://stackoverflow.com/questions/56048952/is-it-possible-to-implement-a-thread-safe-circular-bufffer-that-consists-of-arra

class circular_buffer_t {
private:
    short buffer[BUFFER_SIZE][WIDTH][HEIGHT] = {};
    bool readable_frames[BUFFER_SIZE] = {};
    int read_pos = 0;
    int write_pos = 0;

public:
    circular_buffer_t() {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            readable_frames[i] = false;
        }
    }

    void print_readable_frames() {
        std::cout << "Readable frames: ";
        for (int i = 0; i < BUFFER_SIZE; i++) {
            std::cout << readable_frames[i] << " ";
        }
        std::cout << std::endl;
    }

    void dump_buffer() {
        cv::Mat dumpImg(HEIGHT, WIDTH*BUFFER_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < BUFFER_SIZE; i++) {
            for (int j = 0; j < WIDTH; j++) {
                for (int k = 0; k < HEIGHT; k++) {
                    short px = buffer[i][j][k];
                    int r = ((px & 0b0000111100000000) >> 8) * 16;
                    int g = ((px & 0b0000000011110000) >> 4) * 16;
                    int b = ((px & 0b0000000000001111)) * 16;
                    dumpImg.at<cv::Vec3b>(k, i*WIDTH+j) = cv::Vec3b(b,g,r);
                }
            }
        }
        cv::imshow("Dump", dumpImg);           
    }

    bool empty() {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            if (readable_frames[i]) return false;
        }
        return true;
    }

    int next_read_pos() {
        return (read_pos+1) % BUFFER_SIZE;
    }

    int next_write_pos() {
        return (write_pos+1) % BUFFER_SIZE;
    }

    void incr_read_pos() {
        if (this->empty()) return;
        readable_frames[read_pos] = false;
        read_pos = next_read_pos();
        if (flagVerboseBuffer) {
            std::cout << "Read pos: " << read_pos << std::endl;
            print_readable_frames();
        }
    }

    void incr_write_pos() {
        readable_frames[write_pos] = true;
        write_pos = next_write_pos();
        if (flagVerboseBuffer) {
            std::cout << "Write pos: " << write_pos << std::endl;
            print_readable_frames();
        }
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
        assert(readable_frames[read_pos]);

        return buffer[read_pos][i][j];
    }

    void write(int i, int j, short v) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);
        assert(!readable_frames[write_pos]);

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
    if (flagVerbose) debug_px(px);

    if (flagSpiEnabled) {
        // Préparation des données pour l'envoi via SPI
        unsigned char data[2];
        data[0] = px >> 8; // MSB
        data[1] = px & 0xFF; // LSB
        #ifdef WIRING_PI
        wiringPiSPIDataRW(SPI_CHANNEL, data, 2);
        #endif
    }
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
            buf->write(j,i,pixel);
        }
    }
}

long long now_micros() {
    //see: https://stackoverflow.com/a/2834294
    //from: https://stackoverflow.com/questions/2831841/how-to-get-the-time-in-milliseconds-in-c?answertab=trending#tab-top
    namespace sc = std::chrono;

    auto time = sc::high_resolution_clock::now(); // get the current time

    auto since_epoch = time.time_since_epoch(); // get the duration since epoch

    // I don't know what system_clock returns
    // I think it's uint64_t nanoseconds since epoch
    // Either way this duration_cast will do the right thing
    auto micros = sc::duration_cast<sc::microseconds>(since_epoch);

    long long now = micros.count(); // just like java (new Date()).getTime();
    return now;
}

long long now_nanos() {
        //see: https://stackoverflow.com/a/2834294
    //from: https://stackoverflow.com/questions/2831841/how-to-get-the-time-in-milliseconds-in-c?answertab=trending#tab-top
    namespace sc = std::chrono;

    auto time = sc::high_resolution_clock::now(); // get the current time

    auto since_epoch = time.time_since_epoch(); // get the duration since epoch

    // I don't know what system_clock returns
    // I think it's uint64_t nanoseconds since epoch
    // Either way this duration_cast will do the right thing
    auto nanos = sc::duration_cast<sc::nanoseconds>(since_epoch);

    long long now = nanos.count(); // just like java (new Date()).getTime();
    return now;
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> [-v] [--no-spi] [-g] [-b] [-vb] [--laser-sim]" << std::endl;
        return -1;
    }
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "-v") flagVerbose = true;
            if (std::string(argv[i]) == "--no-spi") flagSpiEnabled = false;
            if (std::string(argv[i]) == "-g") flagGraphicDisp = true;
            if (std::string(argv[i]) == "-b") flagDumpBuffer = true;
            if (std::string(argv[i]) == "-vb") flagVerboseBuffer = true;
            if (std::string(argv[i]) == "--laser-sim") flagLaserSim = true;
        }
    }

    std::cout << "Verbose: " << flagVerbose << std::endl;
    std::cout << "SPI enabled: " << flagSpiEnabled << std::endl;
    std::cout << "Graphic display: " << flagGraphicDisp << std::endl;
    std::cout << "Buffer dump: " << flagDumpBuffer << std::endl;
    std::cout << "Verbose buffer: " << flagVerboseBuffer << std::endl;
    std::cout << "Laser simulation: " << flagLaserSim << std::endl;

    // Initialisation de WiringPi et SPI
    if (flagSpiEnabled) {
        #ifdef WIRING_PI
        if (wiringPiSetup() == -1) {
            std::cerr << "Failed to initialize wiringPi" << std::endl;
            return -1;
        }
        if (wiringPiSPISetup(SPI_CHANNEL, SPI_SPEED) == -1) {
            std::cerr << "Failed to setup SPI" << std::endl;
            return -1;
        }
        #endif
    }

    // open video with OpenCV
    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    //prepare graphical display
    cv::Mat canvas(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    // sync init
    long long idealPxIntervalMicros = (1000000 /*micros*/ / (FPS * WIDTH * HEIGHT));
    long long pxIntervalMicros = idealPxIntervalMicros;
    int cursorX = 0;
    int cursorY = 0;
    auto now = now_micros();
    auto then = now_micros();
    auto nowDisp = now_micros();
    auto thenDisp = now_micros();
    auto videoStart = now_nanos();
    long long ellapsedOneFrameNanos = 0;

    // enslavement
    long long sendStart = now_micros();
    long long sendStop = now_micros();
    bool adjustInterval = false;
    const double PROPORTIONALITY_COEF = 1/2;

    // other initialization
    auto start = std::chrono::system_clock::now();
    bool firstSend = true;
    int nbFrames = 0;
    bool refreshDisp = false;
    bool cachedAllVideo = false;
    std::shared_ptr<circular_buffer_t> frame_buffer = std::make_shared<circular_buffer_t>();


    while(true) {
        now = now_micros();
        nowDisp = now_micros();
        auto ellapsedMicros = now - then;
        if (flagLaserSim) {
            ellapsedOneFrameNanos = (now_nanos() - videoStart) % (1000000000 / FPS);
        }

        if (frame_buffer->next_write_pos_available() && !cachedAllVideo && (ellapsedMicros < (pxIntervalMicros/2) || frame_buffer->empty())) {
            frame_buffer->incr_write_pos();
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Cached all video" << std::endl;
                cachedAllVideo = true;
            } else {
                nbFrames++;
                if (flagVerboseBuffer) std::cout << "Frame " << nbFrames << " being cached." << std::endl;

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
            }
            frame.release();
        } else {
            if (frame_buffer->current_frame_readable()) {  
                if (ellapsedMicros > pxIntervalMicros) {
                    // send
                    if (firstSend) {
                        firstSend = false;
                        start = std::chrono::system_clock::now();
                    }
                    send_px(frame_buffer->read(cursorX,cursorY));
                    if (flagGraphicDisp) {
                        auto px = frame_buffer->read(cursorX,cursorY);
                        int r = ((px & 0b0000111100000000) >> 8) * 16;
                        int g = ((px & 0b0000000011110000) >> 4) * 16;
                        int b = ((px & 0b0000000000001111)) * 16;

                        if (flagLaserSim) {
                            long long wScanLengthNanos = 1000000000 / FPS / HEIGHT / WIDTH;
                            // we find the ellapsed time modulo the spanning time.
                            // Then we use the rule of three to map the position
                            // in the timeframe into a pixel coordinate.
                            int x = ((ellapsedOneFrameNanos % (wScanLengthNanos)) * WIDTH / wScanLengthNanos) % WIDTH;
                            long long hScanLengthNanos = 1000000000 / FPS / HEIGHT;
                            int y = ((ellapsedOneFrameNanos % hScanLengthNanos) * HEIGHT / hScanLengthNanos) % HEIGHT;
                            canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(b,g,r);
                        } else {
                            canvas.at<cv::Vec3b>(cursorY, cursorX) = cv::Vec3b(b,g,r);
                        }
                    }
        
                    // update state
                    int pixelsEllapsed = ellapsedMicros / pxIntervalMicros;
                    if (pixelsEllapsed > 100 && flagVerboseBuffer) std::cout << "skipping " << pixelsEllapsed-1 << " pixels." << std::endl;
                    while (pixelsEllapsed > WIDTH*HEIGHT) {
                        pixelsEllapsed -= (WIDTH*HEIGHT);
                        frame_buffer->incr_read_pos();
                        adjustInterval = true;
                    }

                    auto oldX = cursorX;
                    cursorX = (cursorX + pixelsEllapsed) % WIDTH;
                    
                    //Changed of line?
                    if (cursorX < oldX) {
                        auto oldY = cursorY;
                        cursorY = (cursorY + 1) % 100;

                        //Changed of frame?
                        if (cursorY < oldY) {
                            frame_buffer->incr_read_pos();
                            adjustInterval = true;
                        }
                    }

                    then = now;
                }
            } else if (cachedAllVideo) {
                break;
            }
        }

        if (flagGraphicDisp) {
            refreshDisp = nowDisp - thenDisp > ((1.0/FPS) * 1000000);
            if (refreshDisp) {
                cv::Mat scaledCanvas;
                const int SCALE = 4; 
                cv::resize(canvas, scaledCanvas, cv::Size(SCALE*HEIGHT, SCALE*WIDTH), SCALE, SCALE, cv::INTER_NEAREST);
                cv::imshow("Display", scaledCanvas);
                if (flagDumpBuffer) frame_buffer->dump_buffer();
                cv::waitKey(1); // trigger refresh
                thenDisp = nowDisp;
                canvas.release();
                canvas = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
                scaledCanvas.release();
            }
        }

        if (adjustInterval) {
            // adjust interval based on send speed (enslavement)
            sendStop = now_micros();
            long long sendDuration = (sendStop - sendStart);
            long long averagePxSendDuration = sendDuration / (WIDTH*HEIGHT);
            std::cout << averagePxSendDuration << std::endl;
            long long deltaError = averagePxSendDuration - idealPxIntervalMicros;
            long long adjustedDeltaError = (deltaError * PROPORTIONALITY_COEF);
            if (adjustedDeltaError < 1) adjustedDeltaError = 1;

            if (averagePxSendDuration > idealPxIntervalMicros) {
                pxIntervalMicros = pxIntervalMicros - adjustedDeltaError;
            } else if (averagePxSendDuration < idealPxIntervalMicros) {
                pxIntervalMicros = pxIntervalMicros + adjustedDeltaError;
            }
            if (pxIntervalMicros < 1) pxIntervalMicros = 1;

            sendStart = now_micros();
            adjustInterval = false;
        }
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop - start;
    auto elapsed_time = duration.count();

    std::cout << "Time (s): " << elapsed_time << std::endl;
    double timeMsPerFrame = (elapsed_time / static_cast<double>(nbFrames)) * 1000;
    std::cout << "Per frame (ms): " << timeMsPerFrame << std::endl;
    double timeMsPerPixel = (timeMsPerFrame / static_cast<double>(WIDTH * HEIGHT));
    std::cout << "Per pixel (microseconds): " << timeMsPerPixel*1000 << std::endl;
    if (flagGraphicDisp) cv::waitKey(30000);

    cap.release();
    canvas.release();
    cv::destroyAllWindows();

    return 0;
}
