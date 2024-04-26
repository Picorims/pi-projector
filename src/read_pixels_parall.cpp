#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdlib.h>
#include <regex>

#ifdef WIRING_PI
#include <wiringPi.h>
#include <wiringPiSPI.h>
#endif

#define WIDTH 100
#define HEIGHT 100
#define FPS 10
#define BUFFER_SIZE 10000
const double WIDTH_DOUBLE = static_cast<double>(WIDTH);
const double HEIGHT_DOUBLE = static_cast<double>(HEIGHT);
bool flagVerbose = false;
bool flagSpiEnabled = true;
bool flagGraphicDisp = false;
bool flagDumpBuffer = false;
bool flagVerboseBuffer = false;
bool flagVerboseBufferExtra = false;
bool flagLaserSim = false;
bool flagFullCache = false;
bool flagLoop = false;
bool flagColorOffset = false;
bool flagMirrorVertical = false;
bool flagMirrorHorizontal = false;
int loopCount = 0;

// Initialisation de WiringPi SPI
const int SPI_CHANNEL = 0; // Utilisez le canal 0 de SPI
const int SPI_SPEED = 1000000; // 1 MHz

//offset
const int OFFSET_X_R = 5, OFFSET_Y_R = 0;
const int OFFSET_X_G = 0, OFFSET_Y_G = 4;
const int OFFSET_X_B = -3, OFFSET_Y_B = -3;


// https://stackoverflow.com/questions/56048952/is-it-possible-to-implement-a-thread-safe-circular-bufffer-that-consists-of-arra

class circular_buffer_t {
private:
    short buffer[BUFFER_SIZE][WIDTH][HEIGHT] = {}; // [x][y]
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
        if (flagVerboseBufferExtra) std::cout << "Readable frames: ";
        int count;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            if (readable_frames[i]) count++;
            if (flagVerboseBufferExtra) std::cout << readable_frames[i] << " ";
        }
        if (flagVerboseBufferExtra) std::cout << std::endl;
        std::cout << "Total readable frames: " << count << std::endl;
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

    void writeR(int i, int j, short v) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);
        assert(!readable_frames[write_pos]);

        short currentVal = buffer[write_pos][i][j];
        short eraseRMask = 0xF0FF;
        short newR = (currentVal & eraseRMask) | (v << 8);
        buffer[write_pos][i][j] = newR;
    }

    void writeG(int i, int j, short v) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);
        assert(!readable_frames[write_pos]);

        short currentVal = buffer[write_pos][i][j];
        short eraseRMask = 0xFF0F;
        short newG = (currentVal & eraseRMask) | (v << 4);
        buffer[write_pos][i][j] = newG;
    }

    void writeB(int i, int j, short v) {
        assert(i >= 0 && j >= 0);
        assert(i < WIDTH && j < HEIGHT);
        assert(!readable_frames[write_pos]);

        short currentVal = buffer[write_pos][i][j];
        short eraseRMask = 0xFFF0;
        short newB = (currentVal & eraseRMask) | v;
        buffer[write_pos][i][j] = newB;
    }

    void loop() {
        read_pos = 0;
        for (int i = 0; i < write_pos; i++) {
            readable_frames[i] = true;
        }
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

            if ((OFFSET_X_R == 0 && OFFSET_Y_R == 0 && OFFSET_X_G == 0 && OFFSET_Y_G == 0 && OFFSET_X_B == 0 && OFFSET_Y_B == 0) || !flagColorOffset) {
                short pixel = (r16 << 8) + (g16 << 4) + b16;
                int x = j;
                int y = i;
                if (flagMirrorHorizontal) x = WIDTH-1 - x;
                if (flagMirrorVertical) y = HEIGHT-1 - y;
                buf->write(x, y, pixel);
            } else {
                // Apply offset for R, G, B
                int r_i = std::max(0, std::min(HEIGHT-1, i + OFFSET_Y_R));
                int r_j = std::max(0, std::min(WIDTH-1, j + OFFSET_X_R));
                int g_i = std::max(0, std::min(HEIGHT-1, i + OFFSET_Y_G));
                int g_j = std::max(0, std::min(WIDTH-1, j + OFFSET_X_G));
                int b_i = std::max(0, std::min(HEIGHT-1, i + OFFSET_Y_B));
                int b_j = std::max(0, std::min(WIDTH-1, j + OFFSET_X_B));

                if (flagMirrorHorizontal) {
                    r_j = WIDTH-1 - r_j;
                    g_j = WIDTH-1 - g_j;
                    b_j = WIDTH-1 - b_j;
                }
                if (flagMirrorVertical) {
                    r_i = HEIGHT-1 - r_i;
                    g_i = HEIGHT-1 - g_i;
                    b_i = HEIGHT-1 - b_i;
                }

                short pixel = (r16 << 8) + (g16 << 4) + b16;
                if (i >= OFFSET_Y_R && j >= OFFSET_X_R) {
                    buf->writeR(r_j, r_i, r16);
                }
                if (i >= OFFSET_Y_G && j >= OFFSET_X_G) {
                    buf->writeG(g_j, g_i, g16);
                }
                if (i >= OFFSET_Y_B && j >= OFFSET_X_B) {
                    buf->writeB(b_j, b_i, b16);
                }
            }
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
        std::cerr << "Usage: " << argv[0] << " <video_file_path> [-v] [--no-spi] [-g] [-b] [-vb] [-vbe] [--laser-sim] [--full-cache] [--loop [amount=1]] [--color-offset] [-mv] [-mh]" << std::endl;
        std::cerr << "-v: verbose" << std::endl;
        std::cerr << "--no-spi: disable SPI transmission" << std::endl;
        std::cerr << "-g: show graphical window" << std::endl;
        std::cerr << "-b: show graphical window for the buffer" << std::endl;
        std::cerr << "-vb: verbose for buffering" << std::endl;
        std::cerr << "-vbe: extra verbose for buffering" << std::endl;
        std::cerr << "--laser-sim: laser positioning simulation in graphical window" << std::endl;
        std::cerr << "--full-cache: cache all the video before launching transmission (short videos only)" << std::endl;
        std::cerr << "--loop [n]: loop the video. It will go n times back to start" << std::endl;
        std::cerr << "--color-offset: enable color offset (as defined in the code constants)" << std::endl;
        std::cerr << "-mv: vertical mirroring" << std::endl;
        std::cerr << "-mh: horizontal mirroring" << std::endl;

        return -1;
    }
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "-v") flagVerbose = true;
            else if (std::string(argv[i]) == "--no-spi") flagSpiEnabled = false;
            else if (std::string(argv[i]) == "-g") flagGraphicDisp = true;
            else if (std::string(argv[i]) == "-b") flagDumpBuffer = true;
            else if (std::string(argv[i]) == "-vb") flagVerboseBuffer = true;
            else if (std::string(argv[i]) == "-vbe") flagVerboseBufferExtra = true;
            else if (std::string(argv[i]) == "--laser-sim") flagLaserSim = true;
            else if (std::string(argv[i]) == "--full-cache") flagFullCache = true;
            else if (std::string(argv[i]) == "--loop") {
                flagLoop = true;
                if (i+1 < argc && std::regex_match(argv[i+1], std::regex("[0-9]+"))) {
                    loopCount = std::stoi(argv[i+1]);
                    i++;
                } else {
                    loopCount = 1;
                }
            }
            else if (std::string(argv[i]) == "--color-offset") flagColorOffset = true;
            else if (std::string(argv[i]) == "-mv") flagMirrorVertical = true;
            else if (std::string(argv[i]) == "-mh") flagMirrorHorizontal = true;
            else {
                std::cerr << "unknown flag: " << argv[i] << ", ignoring this flag." << std::endl;
            }
        }
    }

    std::cout << "Verbose: " << flagVerbose << std::endl;
    std::cout << "SPI enabled: " << flagSpiEnabled << std::endl;
    std::cout << "Graphic display: " << flagGraphicDisp << std::endl;
    std::cout << "Buffer dump: " << flagDumpBuffer << std::endl;
    std::cout << "Verbose buffer: " << flagVerboseBuffer << std::endl;
    std::cout << "Verbose buffer Extra: " << flagVerboseBufferExtra << std::endl;
    std::cout << "Laser simulation: " << flagLaserSim << std::endl;
    std::cout << "Full cache: " << flagFullCache << std::endl;
    std::cout << "Loop: " << flagLoop << std::endl;
    std::cout << "Loop count (number of times it triggers a repeat): " << loopCount << std::endl;
    std::cout << "Mirroring - horizontal: " << flagMirrorHorizontal << std::endl;
    std::cout << "Mirroring - vertical: " << flagMirrorVertical << std::endl;

    std::cout << "NOTE: In this version, gamma and aspect ratio are not implemented." << std::endl;

    if (flagLoop && !flagFullCache) {
        std::cerr << "Loop flag requires full cache flag to be set." << std::endl;
        return -1;
    }

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
    auto videoStartNanos = now_nanos();
    auto lastPixelSendStart = now_micros();
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
    int nbPixels = 0;
    bool refreshDisp = false;
    bool cachedAllVideo = false;
    std::shared_ptr<circular_buffer_t> frame_buffer = std::make_shared<circular_buffer_t>();
    int initialLoopCount = loopCount;


    while(true) {
        now = now_micros();
        nowDisp = now_micros();
        auto ellapsedMicros = now - then;
        if (flagLaserSim) {
            ellapsedOneFrameNanos = (now_nanos() - videoStartNanos) % (1000000000 / FPS);
        }

        if (frame_buffer->next_write_pos_available() && !cachedAllVideo && ((now - lastPixelSendStart) < (pxIntervalMicros/10) || frame_buffer->empty() || flagFullCache)) {
            frame_buffer->incr_write_pos();
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Cached all video" << std::endl;
                cachedAllVideo = true;
            } else {
                nbFrames++;
                if (flagVerboseBuffer) {
                    std::cout << "Frame " << nbFrames << " being cached.";
                    std::cout << "\tellapsed since last send (ms): " << (now - lastPixelSendStart) << std::endl;
                }

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
                    lastPixelSendStart = now_micros();
                    if (firstSend) {
                        firstSend = false;
                        start = std::chrono::system_clock::now();
                        if (flagFullCache) videoStartNanos = now_nanos();
                    }
                    send_px(frame_buffer->read(cursorX,cursorY));
                    nbPixels++;
                    if (flagGraphicDisp) {
                        auto px = frame_buffer->read(cursorX,cursorY);
                        int r = ((px & 0b0000111100000000) >> 8) * 16;
                        int g = ((px & 0b0000000011110000) >> 4) * 16;
                        int b = ((px & 0b0000000000001111)) * 16;

                        if (flagLaserSim) {
                            long long wScanLengthNanos = 1000000000 / FPS / HEIGHT;
                            // we find the ellapsed time modulo the spanning time.
                            // Then we use the rule of three to map the position
                            // in the timeframe into a pixel coordinate.
                            double w = WIDTH_DOUBLE;
                            int x = (int) (((ellapsedOneFrameNanos % wScanLengthNanos) * (w / wScanLengthNanos)) /*round to floor through cast*/) % WIDTH;
                            long long hScanLengthNanos = 1000000000 / FPS;
                            double h = HEIGHT_DOUBLE;
                            int y = (int) (((ellapsedOneFrameNanos % hScanLengthNanos) * (h / hScanLengthNanos)) /*round to floor through cast*/) % HEIGHT;
                            canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(b,g,r);
                        } else {
                            canvas.at<cv::Vec3b>(cursorY, cursorX) = cv::Vec3b(b,g,r);
                        }
                    }
        
                    // update state
                    int pixelsEllapsed = ellapsedMicros / pxIntervalMicros;
                    if (pixelsEllapsed > 500 && flagVerboseBuffer) std::cout << "skipping " << pixelsEllapsed-1 << " pixels." << std::endl;
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
                if (flagFullCache && flagLoop && loopCount > 0) {
                    frame_buffer->loop();
                    loopCount--;
                    std::cout << "LOOP (left: " << loopCount << ")" << std::endl;
                } else {
                    break;
                }
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
            if (flagVerboseBuffer) std::cout << "average frame pixel sending duration: " << averagePxSendDuration << std::endl;
            
            //error
            long long deltaError = averagePxSendDuration - idealPxIntervalMicros;
            long long adjustedDeltaError = (deltaError * PROPORTIONALITY_COEF);
            if (adjustedDeltaError < 1) adjustedDeltaError = 1;

            //enslaving
            if (averagePxSendDuration > idealPxIntervalMicros) {
                pxIntervalMicros = pxIntervalMicros - adjustedDeltaError;
            } else if (averagePxSendDuration < idealPxIntervalMicros) {
                pxIntervalMicros = pxIntervalMicros + adjustedDeltaError;
            }
            if (pxIntervalMicros < 1) pxIntervalMicros = 1;

            //reset
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
    double percentSent = static_cast<double>(nbPixels) / (nbFrames*WIDTH*HEIGHT * (initialLoopCount+1)) * 100;
    std::cout << "Sent: " << percentSent << " %" << std::endl;

    //wait 30s at most 
    if (flagGraphicDisp) cv::waitKey(30000);

    cap.release();
    canvas.release();
    cv::destroyAllWindows();

    return 0;
}
