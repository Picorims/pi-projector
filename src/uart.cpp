#include <iostream>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <cstring>
int main() {
    int serialPort;
    if ((serialPort = serialOpen("/dev/serial1", 115200)) < 0) { // Remplacez "/dev/serial0" par votre port série si différent
        std::cerr << "Erreur d'ouverture du port série : " << strerror(errno) << std::endl;
        return 1;
    }

    wiringPiSetup(); // Initialisation de WiringPi

    const char *message = "Hello ESP32\n";
    std::cout << "Envoi du message : " << message;
    serialPuts(serialPort, message);

    serialClose(serialPort); // Fermeture du port série
    return 0;
}
