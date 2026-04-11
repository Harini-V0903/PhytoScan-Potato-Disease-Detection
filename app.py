import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PhytoScan 🌿", layout="centered")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    return tf.keras.models.load_model(os.path.join(BASE_DIR, "potato_model.h5"))

model = load_model()
class_names = ["Early Blight", "Late Blight", "Healthy"]

# ---------------- BACKGROUND + STYLE ----------------
st.markdown("""
<style>

/* Background image */
.stApp {
    background: url("https://images.unsplash.com/photo-1497250681960-ef046c08a56e") no-repeat center center fixed;
    background-size: cover;
}

/* Dark overlay for better contrast */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(20, 60, 20, 0.65); /* darker overlay */
    backdrop-filter: blur(4px);
    z-index: -1;
}

/* Remove white container */
[data-testid="stAppViewContainer"] {
    background: transparent;
}

/* GLOBAL TEXT → WHITE */
html, body, [class*="css"] {
    color: white !important;
    font-weight: 600;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.7); /* glow for visibility */
}

/* Card */
.card {
    background: rgba(0, 0, 0, 0.6); /* darker card */
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    text-align: center;
    color: white;
}

/* Image */
.small-img img {
    border-radius: 10px;
    max-height: 220px;
    object-fit: cover;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌿 PhytoScan</div>', unsafe_allow_html=True)
st.write("")

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader("📤 Browse Image", type=["jpg", "png", "jpeg"])

if uploaded is not None:

    col1, col2 = st.columns([1, 1])

    # ---------------- IMAGE ----------------
    with col1:
        image = Image.open(uploaded)
        st.markdown('<div class="small-img">', unsafe_allow_html=True)
        st.image(image)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREDICTION + SUGGESTION ----------------
    with col2:
        with st.spinner("Analyzing... 🌱"):
            time.sleep(1)
            img = preprocess(image)
            pred = model.predict(img)[0]

        pred_class = class_names[np.argmax(pred)]
        conf = np.max(pred)

        st.markdown(f"""
        <div class="card">
            <h3>{pred_class}</h3>
            <p>Confidence: {conf*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
if pred_class == "Early Blight":
    suggestion_text = "Remove affected leaves and apply fungicide."
elif pred_class == "Late Blight":
    suggestion_text = "Immediate treatment required. Use strong fungicide."
else:
    suggestion_text = "Healthy plant. Maintain proper care."

st.markdown(f"""
<div class="card">
    <p style="color:white; font-size:16px; font-weight:600;">
        🌿 {suggestion_text}
    </p>
</div>
""", unsafe_allow_html=True)
    # ---------------- CLASS SCORES ----------------
st.write("")
st.subheader("📊 Class Scores")
for i in range(len(class_names)):
    st.write(f"{class_names[i]}: {pred[i]*100:.2f}%")