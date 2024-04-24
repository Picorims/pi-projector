#include <iostream>
#include <wiringPi.h>
#include <wiringPiSPI.h>

const int SPI_CHANNEL = 0; // Canal SPI à utiliser (0 ou 1 sur le Raspberry Pi)
const int SPI_SPEED = 1000000; // Vitesse de la communication SPI, ici 1MHz

int main() {
    // Initialisation de WiringPi
    if (wiringPiSetup() == -1) {
        std::cerr << "Erreur d'initialisation de WiringPi" << std::endl;
        return 1;
    }

    // Configuration du canal SPI
    if (wiringPiSPISetup(SPI_CHANNEL, SPI_SPEED) == -1) {
        std::cerr << "Erreur de configuration du canal SPI" << std::endl;
        return 1;
    }

    unsigned char dataToSend = 0xFF; // Donnée à envoyer

    while(true) {
        unsigned char dataReceived = dataToSend; // Préparation de la donnée à envoyer

        // Envoi de la donnée via SPI et réception de la réponse dans le même buffer
        if (wiringPiSPIDataRW(SPI_CHANNEL, &dataReceived, 1) == -1) {
            std::cerr << "Erreur lors de la communication SPI" << std::endl;
            return 1;
        }

        // Affichage de la donnée envoyée pour confirmation
        std::cout << "Donnée envoyée et reçue : 0x" << std::hex << static_cast<int>(dataReceived) << std::endl;

        delay(1000); // Attente d'une seconde avant le prochain envoi
    }

    return 0;
}
