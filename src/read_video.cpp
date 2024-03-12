// #include <opencv2/opencv.hpp>
//#include <opencv4>
#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

bool verbose = false;

/**
 * Print pixel to console
*/
void debug_px(short px) {
    std::cout << std::bitset<16>(px); //bits
    //color if terminal supports it, see https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    std::string reset = "\033[0m";
    int r = ((px & 0b0000111100000000) >> 8) * 16;
    int g = ((px & 0b0000000011110000) >> 4) * 16;
    int b = ((px & 0b0000000000001111)) * 16;
    std::cout << "(" << r << ";" << g << ";" << b << ")";
    std::cout << " \033[48;2;" << r << ";" << g << ";" << b << "m        " /*display size in spaces*/ << reset;
    std::cout << std::endl;
}

/**
 * Send the pixel to the microcontroller.
 * 
 * px: 0000 RRRR GGGG BBBB
*/
void send_px(short px) {
    if (verbose) debug_px(px);
}

int main(int argc, char** argv) {
    // Vérifier les arguments d'entrée
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    std::cout << "launching projector using the following video: " << argv[1];

    if (argc == 3 && argv[2] == std::string("-v")) verbose = true;

    // Ouvrir la vidéo
    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    auto start = std::chrono::system_clock::now();
    int nbFrames = 0;

    // Lire les images frame par frame
    while(true) {
        // simulate wait
        //std::this_thread::sleep_for(std::chrono::milliseconds(500));

        cv::Mat frame;
        cap >> frame; // Lire la frame suivante
        nbFrames++;

        // Vérifier si la frame est vide (fin de la vidéo)
        if(frame.empty()) {
            break;
        }

        // Afficher l'image
        //cv::imshow("Frame", frame);

        // Itérer à travers chaque pixel dans la frame
        for(int i = 0; i < frame.rows; i++) {
            for(int j = 0; j < frame.cols; j++) {
                cv::Vec3b& color = frame.at<cv::Vec3b>(i, j);
                // color[0], color[1], et color[2] sont les composantes B, G, et R du pixel respectivement.
                // Envoi du pixel via I2C...
                int r16 = color[2] / 16;
                int g16 = color[1] / 16;
                int b16 = color[0] / 16;
                short pixel = (r16 << 8) + (g16 << 4) + b16;
                // envoi
                send_px(pixel);
                // simulate wait if not black
                if (verbose) {
                    if (!(r16 == 0 && g16 == 0 && b16 == 0)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    std::cout << "(" << i << ";" << j << ") frame " << nbFrames << ": ";
                }
            }
        }

        // Attendre 30ms ou jusqu'à ce que l'utilisateur appuie sur une touche
        if(cv::waitKey(3000) >= 0) {
            break;
        }
    }

    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = stop - start;
    auto elapsed_time = duration.count();

    std::cout << "time (s): " << elapsed_time << std::endl;
    double timeMsPerFrame = (elapsed_time / (double) nbFrames) * 1000000;
    double timeMsPerPixel = timeMsPerFrame / 10000;
    std::cout << "per frame (microseconds): " << timeMsPerFrame << std::endl;
    std::cout << "per pixel (microseconds): " << timeMsPerPixel << std::endl;

    // Fermer la vidéo et les fenêtres
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
