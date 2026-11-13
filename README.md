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
- **Interactive Web Interface:** Upload images through a Streamlit application.
- **Visual Confidence Scores:** View the top five predictions with confidence bars.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Pillow (PIL)
- Matplotlib
- Streamlit

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
├── app.py
├── cnn_ResNet_18.py
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```


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

## Usage

### Option 1: Streamlit Web Interface

Launch the interactive application:

```bash
python -m streamlit run app.py
```

Streamlit will open the application in your browser, typically at `http://localhost:8501`.

1. Upload a JPG, JPEG, or PNG image.
2. Click **Classify Image**.
3. View the top five predicted ImageNet classes.
4. See the confidence percentage and visual confidence bar for each prediction.

### Option 2: Python Script

You can also run the original classification script:

```bash
python cnn_ResNet_18.py
```

Update the image path inside the script before running it.

The script prints the top five predictions and displays the image using Matplotlib.

**Note:** Pretrained model weights may be downloaded automatically on the first run.

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

- Add support for classifying multiple images.
- Improve input validation and error handling.
- Explore fine-tuning ResNet-18 on a custom dataset.

## Author

**Abidurahman Niazi**

Software Engineering Student | Python, Java , AI & ML

[GitHub Profile](https://github.com/Abid-04)
