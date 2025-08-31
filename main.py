import streamlit as st

# Try multiple ways to import TensorFlow
TENSORFLOW_AVAILABLE = False
tf = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    st.success("TensorFlow imported successfully!")
except ImportError:
    try:
        import tensorflow_cpu as tf
        TENSORFLOW_AVAILABLE = True
        st.success("TensorFlow CPU imported successfully!")
    except ImportError:
        try:
            # Try installing tensorflow-cpu if not available
            import subprocess
            import sys
            st.warning("TensorFlow not found. Attempting to install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.10.0"])
            import tensorflow_cpu as tf
            TENSORFLOW_AVAILABLE = True
            st.success("TensorFlow CPU installed and imported successfully!")
        except Exception as e:
            TENSORFLOW_AVAILABLE = False
            st.error(f"Failed to install TensorFlow: {e}")
            st.error("Please install it manually using: pip install tensorflow-cpu==2.10.0")

try:
    from PIL import Image
    import numpy as np
    import os
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please check if all required packages are installed.")
    st.stop()

# --- Page config ---
st.set_page_config(
    page_title="Malaria Cell Image Classifier",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for professional gradient background and card effect ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e0f7fa 0%, #f6f6f6 100%);
        }
        .main {
            background-color: #fff;
            border-radius: 16px;
            padding: 2rem 2rem 1rem 2rem;
            box-shadow: 0 4px 24px 0 rgba(0,0,0,0.07);
        }
        .stButton>button {
            background-color: #009688;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #00796b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
with st.container():
    st.markdown(
        "<div style='text-align: center;'>"
        "<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJWgpH5kn381ScAmf3_s0mM_AY3bRJI2B40Q&s' width='180' style='margin-bottom: 10px;'/>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='text-align: center; color: #009688;'>Malaria Cell Image Classifier</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #555; font-size: 1.1rem;'>"
        "Upload a blood smear image to check for malaria infection.<br>"
        "<span style='font-size:0.9rem;'>Powered by Deep Learning</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# --- Load the model ---
@st.cache_resource
def load_model():
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow is not available. Cannot load the model.")
        return None
    
    model_path = "malaria_detection by sam (2).keras"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def preprocess_image(img):
    img = img.resize((64, 64))
    img = img.convert("RGB")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- Upload and Prediction Card ---
with st.container():
    st.markdown("### 📤 Upload Cell Image")
    uploaded_file = st.file_uploader(
        "Choose a cell image (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a blood smear image for malaria detection."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if model:
                arr = preprocess_image(image)
                with st.spinner("Analyzing image..."):
                    pred = model.predict(arr)[0][0]
                label = "🦠 <b>Parasitized (Malaria)</b>" if pred > 0.5 else "✅ <b>Uninfected</b>"
                confidence = pred if pred > 0.5 else 1 - pred

                # Show result in a colored box
                if pred > 0.5:
                    st.markdown(
                        f"<div style='background-color:#ffebee;padding:1rem;border-radius:8px;'>"
                        f"<span style='color:#c62828;font-size:1.2rem;'>{label}</span><br>"
                        f"<span style='color:#555;'>Confidence: <b>{confidence:.2%}</b></span>"
                        "</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='background-color:#e8f5e9;padding:1rem;border-radius:8px;'>"
                        f"<span style='color:#2e7d32;font-size:1.2rem;'>{label}</span><br>"
                        f"<span style='color:#555;'>Confidence: <b>{confidence:.2%}</b></span>"
                        "</div>", unsafe_allow_html=True)
            else:
                st.error("Model could not be loaded. Please check the model file.")

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align:center; color:#888; font-size:0.9rem;'>
        Made with ❤️ using Streamlit & TensorFlow<br>
        <a href='https://github.com/samX18-epic' target='_blank'>GitHub: samX18-epic</a>
    </div>
    <div style='text-align:center; color:#888; font-size:0.8rem;'>
      Achieved an accuracy of 96.5% on the test dataset.<br>
    </div>         
""", unsafe_allow_html=True)