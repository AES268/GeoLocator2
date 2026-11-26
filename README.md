# GeoLocator

GeoLocator uses a street-level photograph to predict one of four countries: Japan, Türkiye, the United Arab Emirates, or the United Kingdom. It implements the country-classification component of *Street Objects Detection & Street Location Classification*, a Spring 2025 study that also examined object detection in urban scenes. The study reports **94.21% validation accuracy** for country classification after three epochs.

## Dataset and sampling

The researchers collected more than 80,000 Google Street View images, using the Google Roads API to select road coordinates across major cities in each country. At each sampled coordinate, they collected four views with a 90° field of view. Country labels came from the collection location and were stored as folder names. The classification pool contained approximately 20,000 images per country. The researchers resized the images to 224 × 224 pixels and used no additional augmentation for classification.

## Models

The country classifier processes the whole RGB image with an ImageNet-pretrained ResNet-50. The study trained the classifier using cross-entropy loss, the Adam optimizer, a learning rate of 0.0001, and batches of 24 images. In this repository, [TesterV22.py](TesterV22.py) loads the included `best_model.pth` checkpoint and applies softmax to the four output scores. The classifier takes image pixels as input and runs independently of the object detector.

The study also trained a Faster R-CNN detector with a ResNet-50 backbone to locate street objects. For this task, the researchers manually annotated 500 images and used semi-automated labeling for another 3,500. Augmentation expanded the detection dataset to 20,000 images. The detector distinguished seven object classes plus background. The paper reports a final detector training loss of 0.15, but gives no mean average precision (mAP) score for evaluating detection quality.

## Country-classification results

Table 3 of the study reports the following results for ResNet-50.

| Epoch | Training loss | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.2739 | 0.1634 | 94.03% |
| 2 | 0.1501 | 0.2149 | 91.95% |
| 3 | 0.0900 | 0.1573 | **94.21%** |

The four countries were chosen for their visual distinctiveness. The classifier always assigns one of those four labels, including for photographs taken elsewhere. The paper reports validation results only and does not specify whether all views from each coordinate were kept in the same split. The reported accuracy therefore cannot establish how well the model generalizes to new locations or to a broader set of visually similar countries.

## Source

*Street Objects Detection & Street Location Classification*. Computer Engineering Department, Faculty of Engineering, Pharos University of Alexandria, Spring 2025. Dataset construction and annotation are described in §§3.1–3.1.5, training parameters in Table 2, and classification results in Table 3.
