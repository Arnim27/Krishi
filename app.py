import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input  # Import preprocess_input

# Load the trained model (ensure the correct path to your model file)
model = load_model('resnet_from_scratch.h5')

# Define the class labels (ensure these match the model's training labels)
class_labels = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot", "Tomato_Early_blight",
    "Tomato_healthy", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# App title and description
st.title("Krishi: Plant Disease Detection")
st.write("Upload a plant leaf image, and Krishi will identify if the plant is healthy or has a disease.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image of the plant leaf...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = load_img(uploaded_file, target_size=(128, 128))  # Resize to match model input size
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for model prediction
        image_array = img_to_array(image)  # Convert image to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = preprocess_input(image_array)  # Apply correct preprocessing

        # Predict the class
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100  # Confidence score in percentage

        # Display the prediction and confidence score
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Debug: Display raw prediction probabilities
        st.write(f"Raw prediction probabilities: {prediction[0]}")

        # Optional: Display the prediction probabilities as a bar chart
        st.write("Prediction Probabilities:")
        plt.figure(figsize=(10, 5))
        plt.bar(class_labels, prediction[0], color='skyblue')
        plt.xticks(rotation=90)
        plt.ylabel("Probability")
        st.pyplot(plt)

    except Exception as e:
        # If there was an error, display an error message
        st.error(f"Error occurred while processing the image: {e}")
else:
    st.write("Please upload an image to get a prediction.")
