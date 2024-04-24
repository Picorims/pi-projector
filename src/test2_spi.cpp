#include <pigpio.h>
#include <stdio.h>

int main() {
    if (gpioInitialise() < 0) {
        // Initialisation échouée
        return 1;
    }

    int spiHandle = spiOpen(0, 14000000, 0); // Ouvre le canal SPI 0 à 1MHz

    if (spiHandle < 0) {
        // Échec de l'ouverture de SPI
        gpioTerminate();
        return 2;
    }

    char data[] = {0x01, 0x02, 0x03, 0x04}; // Exemple de données à envoyer
    spiWrite(spiHandle, data, sizeof(data)); // Envoi des données

    spiClose(spiHandle); // Ferme le canal SPI
    gpioTerminate(); // Nettoyage

    return 0;
}

