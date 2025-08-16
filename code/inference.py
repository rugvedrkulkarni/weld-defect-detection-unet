import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from unet_model import UNet

# ===== Preprocessing & Inference =====
def load_image(image_path):
    img = Image.open(image_path).convert("L")
    tensor = TF.to_tensor(img).unsqueeze(0)
    return tensor, img

def infer(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_tensor, raw_img = load_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor))
        pred = (pred > 0.7).float().cpu().squeeze().numpy()


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img, cmap="gray")
    plt.title("Input X-ray")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Defect Mask")
    plt.tight_layout()
    plt.savefig(f"output_mask_{os.path.basename(image_path)}")
    plt.show()

# ===== CLI Wrapper =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on a single X-ray patch image.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained U-Net .pth model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (e.g., W0001/W0001_0001.png)")
    args = parser.parse_args()

    infer(args.model, args.image)
