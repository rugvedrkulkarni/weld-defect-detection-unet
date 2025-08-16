import os
import shutil
import cv2
import albumentations as A
import numpy as np
import random

def img_augmentation(): 
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=15, p=0.5),
    ])           

transform = img_augmentation()

def extract_patches_all_with_val_split(
    w0001_dir, w0002_dir,
    save_root="data",
    patch_size=(320, 480),
    num_augs=3,
    step_size=160,
    val_split=0.2):

    train_img_dir = os.path.join(save_root, "train_images")
    train_mask_dir = os.path.join(save_root, "train_masks")
    val_img_dir = os.path.join(save_root, "val_images")
    val_mask_dir = os.path.join(save_root, "val_masks")

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    patch_data = []
    image_files = sorted([f for f in os.listdir(w0001_dir) if f.endswith(".png")])

    for fname in image_files:
        img_path = os.path.join(w0001_dir, fname)
        mask_path = os.path.join(w0002_dir, fname.replace("W0001", "W0002"))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Warning, Skipped {fname}: image or mask not found")
            continue

        mask = (mask > 127).astype(np.uint8) * 255
        img_height, img_width = img.shape
        cropd_width, cropd_height = patch_size

        for x in range(0, img_width - cropd_width + 1, step_size):
            crop_img = img[0:cropd_height, x : x + cropd_width]
            crop_mask = mask[0:cropd_height, x: x + cropd_width]

            for _ in range(num_augs):
                aug_result = transform(image=crop_img, mask=crop_mask)
                aug_img = aug_result["image"]
                aug_mask = aug_result["mask"]
                patch_data.append((aug_img, aug_mask))

    random.shuffle(patch_data)
    val_count = int(len(patch_data) * val_split)
    train_data = patch_data[val_count:]
    val_data = patch_data[:val_count]

    def save_patches(data, img_dir, mask_dir, prefix):
        for i, (img, mask) in enumerate(data):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            img_path = os.path.join(img_dir, f"{prefix}_{i:04d}.png")
            mask_path = os.path.join(mask_dir, f"{prefix}_{i:04d}.png")
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

    save_patches(train_data, train_img_dir, train_mask_dir, "train")
    save_patches(val_data, val_img_dir, val_mask_dir, "val")

    print(f"Done Saved {len(train_data)} training patches and {len(val_data)} validation patches.")

extract_patches_all_with_val_split(
    w0001_dir="/root/catkin_ws/weld_defect_detection/Welds/Welds/W0001",
    w0002_dir="/root/catkin_ws/weld_defect_detection/Welds/Welds/W0002",
    save_root="/root/catkin_ws/weld_defect_detection/Welds/Welds/data",
    patch_size=(320, 480),
    num_augs=3,
    step_size=160
)
