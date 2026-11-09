# ResNet-18 Image Classifier

A Python-based image classification project using **PyTorch** and a pretrained **ResNet-18 Convolutional Neural Network (CNN)** to identify objects in images.

The application processes an input image, predicts its contents using a model trained on the ImageNet dataset, and displays the top five predicted classes with their confidence scores.

## Features

- **Pretrained ResNet-18:** Uses pretrained ImageNet weights from Torchvision.
- **Image Classification:** Identifies objects across 1,000 ImageNet categories.
- **Top-5 Predictions:** Displays the five most likely classes and their confidence percentages.
- **Automatic Image Preprocessing:** Applies the transformations required by the model.
- **GPU Acceleration:** Supports Apple's Metal Performance Shaders (MPS) when available, with CPU fallback.
- **Result Visualization:** Displays the input image with its most likely predicted class using Matplotlib.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Pillow (PIL)
- Matplotlib

## How It Works

1. Loads a pretrained ResNet-18 model using Torchvision.
2. Selects the MPS device if available, otherwise uses the CPU.
3. Loads an input image and converts it to RGB.
4. Preprocesses the image using the transformations associated with the pretrained model.
5. Performs inference without calculating gradients.
6. Applies Softmax to convert model outputs into probabilities.
7. Displays the top five predictions and their confidence percentages.
8. Visualizes the image with its highest-ranked predicted class.

## Project Structure

```text
ResNet18-Image-Classifier/
├── cnn_ResNet_18.py
├── images/
│   └── trucks.jpg
├── requirements.txt
├── README.md
└── .gitignore
```

The CIFAR-10 dataset folders are not required for this version of the project because the application uses a pretrained ImageNet model.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Abid-04/ResNet18-Image-Classifier.git
cd ResNet18-Image-Classifier
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

## Usage

The application currently loads an image from the `images` directory.

Update the image path in `cnn_ResNet_18.py` to point to the image you want to classify.

For example, if your script uses a relative path:

```python
from pathlib import Path

image_path = Path(__file__).resolve().parent / "images" / "trucks.jpg"
image = Image.open(image_path).convert("RGB")
```

Run the application:

```bash
python cnn_ResNet_18.py
```

The script prints the five most likely classifications with confidence scores and opens a Matplotlib window displaying the input image with its top predicted label.

The pretrained model weights may be downloaded automatically on the first run.

## Example Output

The terminal output follows this format:

```text
Device: mps

Top 5 predictions:
<class name>: <confidence>%
<class name>: <confidence>%
<class name>: <confidence>%
<class name>: <confidence>%
<class name>: <confidence>%
```

The predicted labels and confidence scores depend on the input image. On systems without MPS support, the device will be `cpu`.

## Model Information

This project uses the **ResNet-18** architecture, a convolutional neural network with residual connections.

The model is loaded with pretrained ImageNet weights and used in evaluation mode for inference.

**No additional model training or fine-tuning is performed in this version.**

## Future Improvements

- Build an interactive interface for uploading images.
- Display prediction confidence scores visually.
- Add support for classifying multiple images.
- Improve input validation and error handling.
- Explore fine-tuning ResNet-18 on a custom dataset.

## Author

**Abidurahman Niazi**

Software Engineering Student | Python, Java & AI

[GitHub Profile](https://github.com/Abid-04)
