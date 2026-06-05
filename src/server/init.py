import glob
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


camera_devices = []

def _get_camera_name(dev: str) -> str:
    """Read camera name from sysfs (Linux only). Returns empty string on failure."""
    try:
        video_name = pathlib.Path(dev).name  # e.g. "video2"
        name_path = pathlib.Path(f"/sys/class/video4linux/{video_name}/name")
        return name_path.read_text().strip()
    except Exception:
        return ""

def list_cameras(max_cameras=MAX_CAMERAS):
    global camera_devices
    camera_devices = []

    print("Available cameras:")

    if platform.system() == "Windows":
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera_devices.append(i)
                print(f"[{len(camera_devices) - 1}] Camera {i}")
                cap.release()
    else:
        for dev in sorted(glob.glob("/dev/video*")):
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                camera_devices.append(dev)
                name = _get_camera_name(dev)
                label = f"{dev}  ({name})" if name else dev
                print(f"[{len(camera_devices) - 1}] {label}")
                cap.release()

    return len(camera_devices)


def no_cameras():
    print("No cameras found. Please connect a camera and try again.")


def select_camera_id():
    while True:
        try:
            choice = input("Select camera by number (0, 1, 2...) or quit (q): \n> ")
            if choice.lower() == "q":
                return None
            choice = int(choice)
            if 0 <= choice < len(camera_devices):
                return camera_devices[choice]
            else:
                print(f"\nPlease select an available camera.")
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

