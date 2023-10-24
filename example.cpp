#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Vérifier les arguments d'entrée
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    // Ouvrir la vidéo
    cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    // Lire les images frame par frame
    while(true) {
        cv::Mat frame;
        cap >> frame; // Lire la frame suivante

        // Vérifier si la frame est vide (fin de la vidéo)
        if(frame.empty()) {
            break;
        }

        // Afficher l'image
        cv::imshow("Frame", frame);

        // Itérer à travers chaque pixel dans la frame
        for(int i = 0; i < frame.rows; i++) {
            for(int j = 0; j < frame.cols; j++) {
                cv::Vec3b& color = frame.at<cv::Vec3b>(i, j);
                // color[0], color[1], et color[2] sont les composantes B, G, et R du pixel respectivement.
                // Envoi du pixel via I2C...
            }
        }

        // Attendre 30ms ou jusqu'à ce que l'utilisateur appuie sur une touche
        if(cv::waitKey(30) >= 0) {
            break;
        }
    }

    // Fermer la vidéo et les fenêtres
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
