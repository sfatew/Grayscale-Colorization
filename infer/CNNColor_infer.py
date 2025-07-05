import sys
import argparse
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb

from util import util

from model_build.CNN import colorization


# Define argument parser
parser = argparse.ArgumentParser(description="Image Captioning Inference")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--model_checkpoint", type=str, default=r"model/model_OG_init.pth", help="Path to model checkpoint")
parser.add_argument("--usage", type=str, default="test", help="use for 'test' or 'real'")


def load_model(model_path, device):

    model = colorization()

    state_dict = torch.load(model_path)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.eval().to(device)

    return model

def preprocess_image_test(image_path, image_size):
    """Preprocess the input image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.BICUBIC)
    img = np.array(img) 

    # If img is in (C, H, W) format, convert to (H, W, C)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # Convert (C, H, W) → (H, W, C)

    # Convert RGB → LAB
    img_lab = rgb2lab(img).astype("float32")  # (H, W, 3)

    # Extract L and ab channels
    L = img_lab[:, :, 0] 
    ab = img_lab[:, :, 1:]

    # Convert to PyTorch tensors
    L = torch.tensor(L, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, H, W)
    ab = torch.tensor(ab, dtype=torch.float32).permute(2, 0, 1)  # (2, H, W)

    return {'L': L, 'ab': ab}

def preprocess_image_real(image_path, image_size):
    """Preprocess the input image."""
    transform = transforms.Compose([transforms.ToTensor(),])

    img = Image.open(image_path).convert('L')  # ensures single L channel
    img = img.resize(image_size, Image.BICUBIC)
    # img = np.array(img) 

    return transform(img).unsqueeze(0)  # Add batch dimension

def colorize(model, image_tensor):
    """Generate captions from the model."""

    with torch.no_grad():
        # Encode the input image
        outputs = model(image_tensor)

        #Upsample logits to match 224x224
        logits_upsampled = F.interpolate(outputs, scale_factor=4, mode='bilinear')  # (1, 313, 224, 224)

        # Softmax along the 313 classes
        prob = torch.softmax(logits_upsampled[0].permute(1, 2, 0), dim=-1).cpu().numpy()  # (224, 224, 313)

        pts_in_hull = util.load_pts_in_hull()  # Load pts_in_hull from the file
        ab = util.annealed_mean(prob, pts_in_hull)  # (1, 2, 224, 224)

        L_np = image_tensor[0, 0].cpu().numpy()  # shape: (224, 224)
        lab = np.zeros((224, 224, 3))
        lab[:, :, 0] = L_np
        lab[:, :, 1:] = ab

        rgb = lab2rgb(lab)
    return  rgb

def main(args):
    og_img = args.image_path

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    # Load model and tokenizer
    model_checkpoint = args.model_checkpoint

    image_size = (224,224)

    model = load_model(model_checkpoint, device)

    model_usage = args.usage
    if model_usage == "test":
        out = preprocess_image_test(og_img, image_size)
        image_tensor, ab_image_tensor = out['L'].to(device), out['ab'].to(device)

    elif model_usage == "real":
        image_tensor = preprocess_image_real(og_img, image_size).to(device)
    else:
        raise ValueError("Invalid usage. Use 'test' or 'real'.")


    # Colorizing the image
    img = colorize(model, image_tensor)
    
    og_img = Image.open(og_img).convert("RGB")

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(og_img)

    plt.subplot(122)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)