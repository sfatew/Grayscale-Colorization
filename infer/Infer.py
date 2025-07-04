import subprocess
import sys
import os

INFER_SCRIPTS = {
    "1": ("GAN", "GAN_infer.py"),
    "2": ("CNN", "CNNColor_infer.py"),
    "3": ("VGG-pretrained", "VGGpreColor_infer.py"),
    "4": ("ResUNetGAN", "ResUNetGAN_infer.py"),
}

def main():
    print("Choose the inference model:")
    for k, (name, _) in INFER_SCRIPTS.items():
        print(f"{k}: {name}")
    choice = input("Enter your choice (number): ").strip()
    if choice not in INFER_SCRIPTS:
        print("Invalid choice.")
        return

    script_name = INFER_SCRIPTS[choice][1]
    image_path = input("Enter path to input image: ").strip()
    model_checkpoint = input("Enter path to model checkpoint (or leave blank for default): ").strip()
    usage = input("Usage mode ('test' or 'real', default 'real'): ").strip() or "real"

    # Check if model_checkpoint exists if provided
    if model_checkpoint and not os.path.isfile(model_checkpoint):
        print(f"Error: The model checkpoint '{model_checkpoint}' does not exist or is not a file.")
        return

    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path, "--image_path", image_path, "--usage", usage]
    if model_checkpoint:
        cmd += ["--model_checkpoint", model_checkpoint]

    print("\nRunning:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: The selected model checkpoint is not compatible with the script '{script_name}'.")
        print("Details:", e)
        print("Please check that you selected the correct checkpoint for this model.")

if __name__ == "__main__":
    main()