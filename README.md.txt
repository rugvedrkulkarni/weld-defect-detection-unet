# Weld Defect Detection — U-Net

Short description: semantic segmentation of weld defects from X-ray images using a U-Net model.

## How to Run (quick start)
```bash
# create & activate env (example)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# install deps
pip install -r requirements.txt

# training (example)
python code/train.py

# inference (example)
python code/inference.py --model unet_model.pth --image path/to/your_image.png
