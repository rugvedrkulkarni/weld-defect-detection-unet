import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from unet_model import UNet


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# ===== Dataset =====
class WeldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


# ===== Transforms =====
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(160, 240),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(160, 240),
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ])


def check_accuracy(loader, model, device="cpu"):
    model.eval()
    correct, total = 0, 0
    dice_total = 0
    valid_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  

            preds = torch.sigmoid(model(x))
            preds_bin = (preds > 0.5).float()

            correct += (preds_bin == y).sum()
            total += torch.numel(preds_bin)

            for pred, target in zip(preds_bin, y):
                if target.sum() > 0:
                    intersection = (pred * target).sum()
                    dice_score = (2 * intersection) / (pred.sum() + target.sum() + 1e-8)
                    dice_total += dice_score
                    valid_batches += 1

    acc = 100 * correct / total
    mean_dice = dice_total / valid_batches if valid_batches > 0 else 0.0

    print(f"Accuracy: {acc:.2f}%, Dice Score: {mean_dice:.4f}")
    model.train()


# ===== Main Training Loop =====
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    loss_fn = lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_ds = WeldDataset("../data/train_images", "../data/train_masks", transform=get_transforms(True))
    val_ds = WeldDataset("../data/val_images", "../data/val_masks", transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    for epoch in range(30):
        loop = tqdm(train_loader)
        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/30]")
            loop.set_postfix(loss=loss.item())

        check_accuracy(val_loader, model, device)

    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved as unet_model.pth")

if __name__ == "__main__":
    main()
