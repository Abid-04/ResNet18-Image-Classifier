
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

#will use mps GPU if it is available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


#Loading the pretrained ResNet18(since ResNet_18 is already pretarined, we won´t need to train our model from scratch)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).to(device)
model.eval()


# ResNet_18 is trained with 10000 dfferent classes. common animals, Vehicles & Transport , food and drinks, Everyday Tech & Electronics, Home & Office Objects,
#- and sports are invlolved in those classes.
image = Image.open(
    "/Users/abid/Python/cnn_python/images/trucks.jpg"
).convert("RGB")



preprocess = weights.transforms()

image_tensor = preprocess(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)


top_probs, top_indices = torch.topk(
    probabilities,
    k=5,
    dim=1
)


classes = weights.meta["categories"]

print("Top 5 predictions:")

for i in range(5):
    index = top_indices[0][i].item()
    confidence = top_probs[0][i].item() * 100

    print(f"{classes[index]}: {confidence:.2f}%")


predicted_index = top_indices[0][0].item()
predicted_class = classes[predicted_index]

plt.imshow(image)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()
