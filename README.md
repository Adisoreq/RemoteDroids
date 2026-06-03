# Zdalne Droidy

Zdalnie sterowane gestami dłoni, autonomiczne roboty.
Projekt łączy w sobie elementy elektroniki, robotyki i informatyki.

## Komponenty

### Serwer (Python)

- Platforma: Raspberry Pi / PC
- Kamera: Raspberry Pi AI 3
- Rozpoznawanie gestow: MediaPipe Tasks

### Klient (ESP32)

- Szkic Arduino: `src/client/RDClient.ino`
- Definicje pinow: `src/client/Pins.h`

## Jak zacząć

### Uruchomienie serwera

Przejdź [**tutaj**](docs/guides/setup.md), aby uzyskać więcej informacji.

## Profile zaleznosci

- `requirements.txt` - bazowe zaleznosci runtime
- `requirements-dev.txt` - zaleznosci developerskie (bazowe + narzedzia)
- `requirements-prod.txt` - zaleznosci pod wdrozenie (bazowe, wersje do walidacji na Raspberry Pi)

## Raspberry Pi (prod)

Przykladowy plik uslugi systemd znajduje sie w `scripts/systemd/remote-droids-server.service`.
Szczegoly konfiguracji i uruchomienia: [docs/guides/setup.md](docs/guides/setup.md).

## Wymagania

- **Serwer**
  - Python (3.13+)
  - pip
- **Klient**
  - Arduino IDE
  - esp32 (by Espressif Systems)