# Weld Defect Detection using U-Net

This project applies a **U-Net deep learning model** to detect and segment weld defects from X-ray images. It demonstrates how image segmentation can be used in non-destructive testing for quality control.

---

## Project Files
- `train.py` — trains the U-Net model  
- `inference.py` — runs predictions on test images  
- `augmentation.py` — performs data augmentation  
- `unet_model.py` — defines the U-Net architecture  

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Train the model:
python code/train.py

3. Run inference:
python code/inference.py --model_path saved_model.pth --image_path data/test.png

## Results
<img width="1638" height="535" alt="results" src="https://github.com/user-attachments/assets/9cc8fe6d-564d-4272-9970-b11c7f899665" />


## Tools Used
Python, PyTorch ,Albumentations, OpenCV, NumPy, Matplotlib

## Dataset: 
GDXray weld defect images
