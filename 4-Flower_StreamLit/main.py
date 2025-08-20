import streamlit as st
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

# Class labels (ensure the order matches your trained model)
FLOWER_CLASSES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

# Load the model only once
@st.cache_resource
def get_model():
    return tf.keras.models.load_model("my_model.keras")

model = get_model()

# App title and description
st.header("🌻 Flower Image Classifier")
st.markdown("Upload a flower photo and the model will try to identify its type.")

# Upload section
file = st.file_uploader("Upload a flower image", type=["png", "jpg", "jpeg"])

if file:
    # Display uploaded picture
    st.image(file, caption="Your Uploaded Image", width=300)

    # Convert to model input
    processed_img = load_img(file, target_size=(128, 128))  # adjust if needed
    processed_img = img_to_array(processed_img) / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_index = np.argmax(prediction)
    flower_name = FLOWER_CLASSES[predicted_index]
    confidence = float(np.max(prediction) * 100)

    # Show result
    st.subheader("Model Output:")
    st.write(f"🌼 Predicted flower: **{flower_name}**")
    st.write(f"📊 Confidence: {confidence:.2f}%")
