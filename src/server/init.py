import os
import pathlib
import subprocess
import sys

files = []
selected_model = ""

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def list_files_in_directory(directory: str):
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

def start_server():
    path = pathlib.Path(__file__).parent.parent.parent / "src" / "server" / "recognizer.py"
    command = [sys.executable, str(path), "--model", str(selected_model)]
    subprocess.run(command, check=False)

if __name__ == "__main__":
    list_models()

    if (len(files) == 0):
        no_models()

    else:
        selected_model = select_model()

        if selected_model is None:
            clear()
            exit(0)

        print(f"Selected model: {selected_model}\n")
        start_server()

