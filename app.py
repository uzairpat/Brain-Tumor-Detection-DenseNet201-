import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# Load the pre-trained model and adjust for classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '../brain_tumor/densenet201_brain_tumor.pth'

# Check the correct number of classes (based on your trained model)
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']  # Updated to match model (4 classes)

# Load pre-trained DenseNet201 model
model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features

# Ensure classifier matches the saved model structure
model.classifier = nn.Linear(num_features, len(class_names))  # 4 output neurons

# Load trained model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title('Brain Tumor Classification')

st.write("Upload an image to classify the type of brain tumor.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open image using PIL
    image = Image.open(uploaded_image)

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.Softmax(dim=1)(output)  # Convert to probabilities
        _, predicted_class = torch.max(probabilities, 1)  # Get highest probability class

    # Display results
    tumor_class = class_names[predicted_class.item()]
    confidence = probabilities.max().item() * 100

    st.write(f"**Predicted Tumor Class:** {tumor_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
