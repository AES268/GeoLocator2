# GeoLocator

GeoLocator uses a street-level photograph to predict one of four countries: Japan, Türkiye, the United Arab Emirates, or the United Kingdom. We built it as the country classifier for our Spring 2025 study, *Street Objects Detection & Street Location Classification*, which also examined object detection in urban scenes. Our classifier reached **94.21% validation accuracy** for country classification after three epochs.

## Run the image classifier on Windows

The repository includes our trained model, `best_model.pth`, and example photos in [Test Images](Test%20Images/).

1. Install Python 3.10 with Tkinter. Open PowerShell in the folder containing `GeoLocator.slnx`, create a virtual environment, and install the dependencies.

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install torch torchvision pillow matplotlib piexif
   ```

2. **Work on copies of your photos.** The script writes the predicted country to the EXIF `ImageDescription` field and saves over the input file as JPEG. A PNG can end up containing JPEG data while keeping its `.png` extension. To try one sample, copy it into a separate folder.

   ```powershell
   New-Item -ItemType Directory -Force .\demo-images | Out-Null
   Copy-Item -LiteralPath '.\Test Images\Japan (1).jpg' -Destination .\demo-images
   ```

3. Launch the classifier from the project folder so it can find the checkpoint.

   ```powershell
   .\.venv\Scripts\python.exe .\TesterV22.py
   ```

4. Select `demo-images` in the folder picker. For each JPG, JPEG, or PNG directly inside the folder, the script prints the predicted country and confidence, displays the photo for about two seconds, and writes the country label to the copy. It uses CUDA when available and otherwise runs on the CPU.

If you use Visual Studio, open `GeoLocator.slnx`, select the Python environment with the dependencies installed, and set `TesterV22.py` as the startup file.

## Dataset and sampling

We downloaded more than 80,000 street-level images through the Google Street View API. We used the Google Roads API to select road coordinates across major cities in each country and collected four views per coordinate, each with a 90° field of view. We labeled each image by its collection country and stored it in the corresponding country folder. The classification pool contained approximately 20,000 images per country. We resized the images to 224 × 224 pixels and used no additional augmentation for classification.

## Models

The country classifier processes the whole RGB image with an ImageNet-pretrained ResNet-50. We trained the classifier using cross-entropy loss, the Adam optimizer, a learning rate of 0.0001, and batches of 24 images. In this repository, [TesterV22.py](TesterV22.py) loads the included `best_model.pth` checkpoint and applies softmax to the four output scores. The classifier takes image pixels as input and runs independently of the object detector.

We also trained a Faster R-CNN detector with a ResNet-50 backbone to locate street objects. For this task, we manually annotated 500 images and used semi-automated labeling for another 3,500. We augmented the detection dataset to 20,000 images. The detector distinguished seven object classes plus background. We recorded a final detector training loss of 0.15. We did not report mean average precision (mAP) for detection.

## Country-classification results

We recorded the following ResNet-50 results in Table 3 of our paper.

| Epoch | Training loss | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.2739 | 0.1634 | 94.03% |
| 2 | 0.1501 | 0.2149 | 91.95% |
| 3 | 0.0900 | 0.1573 | **94.21%** |

We chose these four countries for their visual distinctiveness. The classifier always assigns one of those four labels, including for photographs taken elsewhere. We reported validation results only. Our paper does not specify whether we kept all views from each coordinate in the same split. The reported accuracy therefore cannot establish how well the model generalizes to new locations or to a broader set of visually similar countries.

## Project files

- `TesterV22.py` opens the folder picker and runs inference using the included checkpoint.
- `TesterV2.py` runs an earlier version of the classifier with a hard-coded image folder.
- `Trainer V12.py` contains an unfinished training experiment.
- `WeightsChecker.py` compares the weights in two checkpoint files.
- `Test Images/` contains sample photos from all four countries.

If you train a replacement checkpoint, preserve the classifier’s output order: Japan, Türkiye, United Arab Emirates, United Kingdom.

## Source

*Street Objects Detection & Street Location Classification*. Computer Engineering Department, Faculty of Engineering, Pharos University of Alexandria, Spring 2025.
