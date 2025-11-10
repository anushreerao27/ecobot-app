import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- Page Setup ---
st.set_page_config(page_title="♻️ Ecobot - Smart Waste Sorter", layout="wide")
st.title("♻️ Ecobot - Smart Waste Sorter")
st.markdown("Upload an image or take a picture of waste, and Ecobot will classify it!")

# --- Download Model from Google Drive if Not Exists ---
MODEL_FILE = "ecobot_updated.keras"
FILE_ID = "1RP70xfPI9Q_VMnpjIzD8kzOgJyatgOlg"  # your Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_FILE):
    st.write("📦 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_FILE)

# --- Image Input: Camera or File Upload ---
st.write("📸 Capture or upload an image of waste")
image_source = st.radio("Choose image source:", ["Take Photo", "Upload File"])

if image_source == "Take Photo":
    uploaded_file = st.camera_input("Capture Image")
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)  # limit image width

    # Preprocess image for model
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # --- Handle multi-class prediction safely ---
    labels = ["Biodegradable", "Non-Biodegradable", "E-Waste"]
    if prediction.shape[-1] == len(labels):
        class_index = np.argmax(prediction[0])
        label = labels[class_index]
    else:
        # Fallback for binary or unexpected outputs
        label = labels[0] if prediction[0][0] > 0.5 else labels[1]

    # --- Fun Display with Emojis and Colors ---
    emoji_map = {
        "Biodegradable": "🥦🍃",
        "Non-Biodegradable": "🧴🛢️",
        "E-Waste": "💻📱"
    }
    color_map = {
        "Biodegradable": "green",
        "Non-Biodegradable": "red",
        "E-Waste": "orange"
    }

    st.markdown(
        f"<h2 style='color:{color_map[label]};'>Prediction: {label} {emoji_map[label]}</h2>", 
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Ecobot | Smart Waste Classification")
