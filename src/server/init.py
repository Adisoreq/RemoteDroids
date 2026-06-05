import os
import pathlib
import platform
import subprocess
import sys
import cv2


files = []
selected_model = ""
selected_camera_id = 0
MAX_CAMERAS = 10


def clear():
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def list_files_in_directory(directory: str | pathlib.Path):
    """List all files in the given directory recursively."""
    try:
        base_path = pathlib.Path(directory)
        i = 1
        for file_path in sorted(base_path.rglob("*")):
            if file_path.is_file():
                print(f"[{i}]\t{file_path.relative_to(base_path)}")
                files.append(file_path)
                i += 1
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
    except PermissionError:
        print(f"Permission denied for directory '{directory}'.")


def list_models():
    clear()
    current_directory = pathlib.Path(__file__).parent.parent.parent / "assets" / "models"
    print(f"Available models ({current_directory}):")
    list_files_in_directory(current_directory)
    print()


def no_models():
    print("No models found. Please add .task files to the 'assets/models' directory.")


def select_model():
    while True:
        try:
            choice = input("Select a model by number (1, 2, 3...) or quit (q): \n> ")
            if choice.lower() == "q":
                return None
            choice = int(choice)
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"\nPlease select available model.")
        except ValueError:
            print("\nInvalid input. Please enter a number.")


def list_cameras(max_cameras=MAX_CAMERAS):
    backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_DSHOW
    
    print("Available cameras:")
    found = 0
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            print(f"[{i}] Camera {i}")
            cap.release()
            found += 1
    
    return found


def no_cameras():
    print("No cameras found. Please connect a camera and try again.")


def select_camera_id():

    while True:
        try:
            choice = input("Select camera ID (0, 1, 2...) or quit (q): \n> ")
            if choice.lower() == "q":
                return None
            choice = int(choice)
            if choice >= 0:
                return choice
            else:
                print(f"\nPlease select valid camera ID.")
                
        except ValueError:
            print("\nInvalid input. Please enter a number.")


def start_server():
    path = pathlib.Path(__file__).parent.parent.parent / "src" / "server" / "recognizer.py"
    command = [sys.executable, str(path), "--model", str(selected_model), "--camera-id", str(selected_camera_id)]
    subprocess.run(command, check=False)


if __name__ == "__main__":
    list_models()

    if len(files) == 0:
        no_models()

    selected_model = select_model()

    if selected_model is None:
        clear()
        exit(0)

    print(f"Selected model: {selected_model}\n")

    if list_cameras() == 0:
        no_cameras()
        exit(1)
        
    selected_camera_id = select_camera_id()

    if selected_camera_id is None:
        clear()
        exit(0)

    print(f"Selected camera ID: {selected_camera_id}\n")
    
    start_server()

