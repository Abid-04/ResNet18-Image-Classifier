
import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon=""
)

st.title(" AI Image Classifier")
st.write("Upload an image and let ResNet18 identify it.")

# Load the pretrained model only once
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()

# uploading the image from here
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Classify Image"):
        # transform the image for resnet_18
        preprocess = weights.transforms()
        input_tensor = preprocess(image).unsqueeze(0)

        # Run the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output[0], dim=0)

        # Find the 5 highest-scoring classes
        scores, indices = torch.topk(probabilities, 5)

        st.subheader("Top 5 Predictions")

        for score, index in zip(scores, indices):
            label = weights.meta["categories"][index.item()]
            confidence = score.item()

            st.write(f"**{label}** — {confidence:.2%}")
            st.progress(confidence)
