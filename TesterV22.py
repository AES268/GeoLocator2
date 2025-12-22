import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import piexif
import os
import tkinter as tk
from tkinter import filedialog

# 1. SETUP: Select Directory via GUI
# We hide the main root window so only the dialog appears
root = tk.Tk()
root.withdraw()

print("Please select the folder containing your images...")
test_image_dir = filedialog.askdirectory(title="Select Image Folder")

# Check if user cancelled the dialog
if not test_image_dir:
    print("No folder selected. Exiting script.")
    exit()

print(f"Selected Folder: {test_image_dir}")

# 2. SETTINGS
num_classes = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'best_model.pth'
class_names = ['Japan', 'Türkiye', 'United Arab Emirates', 'United Kingdom']

# 3. LOAD MODEL
print("Loading model...")
try:
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # Forces translation to CPU if CUDA is not available on this machine
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find '{model_path}'. Make sure it is in the same folder as this script.")
    exit()

# 4. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5. HELPER FUNCTIONS
def display_image_with_title(image_path, title):
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(2)  # Show for 2 seconds
        plt.close()
    except Exception as e:
        print(f"Could not display image: {e}")

def write_country_to_metadata(image_path, country):
    try:
        img = Image.open(image_path)

        # Load existing EXIF data if available, else start from empty dict
        try:
            exif_dict = piexif.load(img.info["exif"])
        except (KeyError, piexif.InvalidImageDataError, TypeError):
            # TypeError added because sometimes piexif fails on NoneType info
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Set country in ImageDescription (ID = 270)
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = country.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)

        # Re-save image with new EXIF data (must be JPEG!)
        img.save(image_path, "jpeg", exif=exif_bytes)
        print(f"   [Metadata Saved]")
    except Exception as e:
        print(f"   [Metadata Error] Failed to write for {os.path.basename(image_path)}: {e}")

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        predicted_country = class_names[pred.item()]
        confidence = conf.item() * 100
        print(f"{os.path.basename(image_path)} ➔ {predicted_country} ({confidence:.2f}%)")

        display_image_with_title(image_path, f"{predicted_country} ({confidence:.1f}%)")
        write_country_to_metadata(image_path, predicted_country)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")

# 6. MAIN LOOP
# Predict all images in selected folder
files_found = False
for img_name in os.listdir(test_image_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        files_found = True
        img_path = os.path.join(test_image_dir, img_name)
        predict_image(img_path)

if not files_found:
    print("No JPG or PNG images found in the selected folder.")