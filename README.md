# Zdalne droidy

Project by Adrian Piznal

## Elementy

### 1. Serwer

- Raspberry PI
- Kamerka
- Model AI

### 2. Klient

- Zdalnie sterowane robociki
- ESP32

## Szybki start (serwer)

1. Aktywuj virtualenv:
	- PowerShell: `./ZdalneDroidy/Scripts/Activate.ps1`
2. Upewnij sie, ze masz pakiety: `mediapipe`, `opencv-contrib-python`, `numpy`.
3. Pobierz model Gesture Recognizer (`gesture_recognizer.task`) i wstaw do `server/models/`.
4. Uruchom:
	- `python server/src/recogniser.py`

Skrypt otwiera podglad z kamery i pokazuje nazwe wykrytego gestu.
Nacisnij `q`, aby zakonczyc.

### Parametry uruchomienia

- `--model` - sciezka do pliku `.task`
- `--camera-id` - indeks kamery (domyslnie `0`)
- `--min-score` - minimalna pewnosc predykcji (domyslnie `0.5`)

Przyklad:

`python server/src/recogniser.py --camera-id 1 --min-score 0.6`