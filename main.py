import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import pandas as pd
import zipfile

# Unzip model if not already extracted
if not os.path.exists("normal_cnn_model.h5"):
    with zipfile.ZipFile("normal_cnn_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Agromax AI",
    page_icon="🌿",
    layout="wide",
)

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.title("🌿 Agromax AI")

language = st.sidebar.selectbox("🌍 Language", ["English", "Hindi"])
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- DARK MODE ----------------
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---------------- MODEL CONFIG ----------------
MODEL_PATH = "normal_cnn_model.h5"
FILE_ID = "1Y7ubkH1xL8Dv81WMjdLTUl8ZIZ94eRUc"

CLASS_NAMES = ["Healthy", "Bacterial Spot", "Leaf Rust"]

RECOMMENDATIONS = {
    "Healthy": {
        "English": "Your plant is healthy. Maintain proper irrigation and sunlight.",
        "Hindi": "आपका पौधा स्वस्थ है। उचित सिंचाई और धूप बनाए रखें।"
    },
    "Bacterial Spot": {
        "English": "Apply copper-based fungicide and remove infected leaves.",
        "Hindi": "कॉपर आधारित फंगीसाइड का उपयोग करें और संक्रमित पत्तियां हटाएं।"
    },
    "Leaf Rust": {
        "English": "Improve air circulation and apply recommended treatment spray.",
        "Hindi": "हवा का प्रवाह बढ़ाएं और उचित स्प्रे का उपयोग करें।"
    }
}

# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI Model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("Navigation", ["Home", "Predict", "About", "Contact"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🌿 Agromax AI")
    st.subheader("Smart Leaf Disease Detection System")

    if language == "Hindi":
        st.write("AI आधारित फसल रोग पहचान प्रणाली।")
    else:
        st.write("AI-powered crop disease detection platform.")

# ---------------- PREDICT ----------------
elif page == "Predict":

    st.title("🔍 Leaf Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing with AI..."):
            time.sleep(1)

            img = image.resize((128, 128))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100

            predicted_class = CLASS_NAMES[predicted_index]

        # ---------------- RESULTS ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))

        # ---------------- PROBABILITY CHART ----------------
        with col2:
            df = pd.DataFrame({
                "Disease": CLASS_NAMES,
                "Probability (%)": prediction[0] * 100
            })
            st.bar_chart(df.set_index("Disease"))

        # ---------------- AI EXPLANATION ----------------
        st.subheader("🧠 AI Explanation")

        if language == "Hindi":
            st.write("मॉडल पत्तियों के रंग, बनावट और धब्बों के आधार पर भविष्यवाणी करता है।")
        else:
            st.write("The model analyzes leaf color patterns, texture variations, and spot formations to predict disease.")

        # ---------------- FARMER RECOMMENDATION ----------------
        st.subheader("🌾 Farmer Recommendation")
        st.write(RECOMMENDATIONS[predicted_class][language])

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("About Agromax AI")

    st.write("""
    Agromax AI is a next-generation smart agriculture solution 
    powered by Convolutional Neural Networks.
    
    Features:
    - Real-time AI prediction
    - Multi-language support
    - Confidence scoring
    - Farmer recommendations
    - Cloud deployment
    """)

# ---------------- CONTACT ----------------
elif page == "Contact":
    st.title("📩 Contact Us")

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")

    if st.button("Send Message"):
        st.success("Thank you! Our team will contact you soon.")

