# GeoLocator

GeoLocator is a small image-classification project that predicts which of four countries a photo is associated with: Japan, Türkiye, the United Arab Emirates, or the United Kingdom. It uses a ResNet-50 model with PyTorch.

## Run the image classifier

1. Install Python 3.10 or newer and the project dependencies:

   ```powershell
   python -m pip install torch torchvision pillow matplotlib piexif
   ```

2. Place a compatible trained model named `best_model.pth` in the project folder. Model weights are not included in this repository.
3. From the project folder, run:

   ```powershell
   python TesterV22.py
   ```

4. Select a folder when prompted. The script predicts supported JPG and PNG images, prints each predicted country and confidence, and briefly displays each image.

The script writes each prediction to the image's EXIF `ImageDescription` field. It rewrites the selected image as JPEG, so run it on copies of your images; PNG files are also rewritten as JPEG data.

## Project files

- `TesterV22.py` — folder picker, image classification, display, and metadata writing.
- `TesterV2.py` — an earlier classifier that uses a hard-coded image folder.
- `Trainer V12.py` — an experimental training script.
- `WeightsChecker.py` — compares two model checkpoint files.
- `Test Images/` — sample images for the four classes.

The model uses four output classes in this order: Japan, Türkiye, United Arab Emirates, United Kingdom. Use training data arranged in class folders if preparing replacement weights, and make sure the model architecture and class order match the inference script.
