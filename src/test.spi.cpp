#include <iostream>
#include <unistd.h>
#include <wiringPiSPI.h>
#include <map>

using namespace std;

const int CHANNEL = 0; // Vérifiez votre canal
const int SPI_SPEED = 500000; // 500 kHz

// Mappage des caractères à leur représentation pour l'affichage
map<char, unsigned char> charMap = {
    {'A', 0x77}, // exemple : 'A' à 0x77
    {'B', 0x7F}, // exemple : 'B' à 0x7F
    // Ajoutez d'autres caractères ici
};

void sendChar(char c, int position) {
    unsigned char buffer[2];
    buffer[0] = position; // Position sur l'affichage
    buffer[1] = charMap[c]; // Donnée à envoyer

    wiringPiSPIDataRW(CHANNEL, buffer, 2);
}

int main() {
    string name = "AB"; // Le nom à envoyer

    int fd = wiringPiSPISetup(CHANNEL, SPI_SPEED);
    if (fd == -1) {
        cout << "Failed to init SPI communication." << endl;
        return -1;
    }

    for (int i = 0; i < name.length(); ++i) {
        sendChar(name[i], 0x7B + i); // Envoyer chaque caractère
        sleep(1); // Délai entre les caractères
    }

    return 0;
}
