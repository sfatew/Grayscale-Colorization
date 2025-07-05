import sys
import argparse
from pathlib import Path
# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from model_build.ResUNetGAN import build_res_unet

def load_generator(model_path, device):
    """Load the pre-trained generator model."""
    generator = build_res_unet()
    state_dict = torch.load(model_path, map_location=device)
    
    # Filter the keys if they were prefixed like 'generator.'
    generator_state_dict = {k.replace("net_G.", ""): v for k, v in state_dict.items() if k.startswith("net_G.")}
    generator.load_state_dict(generator_state_dict)
    generator.eval()
    return generator

def preprocess_image(image_path, image_size, mode="real"):
    """Preprocess the input image for inference."""
    img = Image.open(image_path).convert("RGB" if mode == "test" else "L")
    img = img.resize(image_size, Image.BICUBIC)
    
    if mode == "test":
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        L = img_lab[..., 0] / 50. - 1.  # Normalize L channel
        ab = img_lab[..., 1:] / 110.    # Normalize ab channels
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        ab = torch.tensor(ab, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        return {'L': L, 'ab': ab}
    else:
        img = np.array(img)
        img = img / 50. - 1.
        L = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return L

def colorize_image(generator, L_tensor):
    """Generate the colorized image using the generator."""
    with torch.no_grad():
        fake_ab = generator(L_tensor)  # Predict ab channels
        L = (L_tensor[0, 0].cpu().numpy() + 1.) * 50.  # Denormalize L channel
        ab = fake_ab[0].permute(1, 2, 0).cpu().numpy() * 110.  # Denormalize ab channels
        lab = np.zeros((L.shape[0], L.shape[1], 3))
        lab[..., 0] = L
        lab[..., 1:] = ab
        rgb = lab2rgb(lab)  # Convert LAB to RGB
        return rgb

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.model_checkpoint, device)
    image_size = (256, 256)

    if args.usage == "test":
        data = preprocess_image(args.image_path, image_size, mode="test")
        L_tensor = data['L'].to(device)
    elif args.usage == "real":
        L_tensor = preprocess_image(args.image_path, image_size, mode="real").to(device)
    else:
        raise ValueError("Invalid usage. Use 'test' or 'real'.")

    # Generate the colorized image
    colorized_img = colorize_image(generator, L_tensor)

    # Display the original and colorized images
    original_img = Image.open(args.image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Colorized Image")
    plt.imshow(colorized_img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Colorization Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_checkpoint", type=str, default= f"model/ResUNet_GAN.pth", help="Path to generator checkpoint")
    parser.add_argument("--usage", type=str, default="real", choices=["test", "real"], help="Usage mode: 'test' or 'real'")
    args = parser.parse_args()
    main(args)