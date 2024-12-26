import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Navigation configuration (this must be the first Streamlit command)
st.set_page_config(page_title="Klasifikasi Warna Matos Fashion", layout="wide")

# URL Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1bpm2Gp_qVqsBIMHw-kRlzHX0e3QNie70"
MODEL_PATH = "color_model.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model berhasil diunduh!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Map label indices to color names
label_map = {0: 'Merah', 1: 'Kuning', 2: 'Biru', 3: 'Hitam', 4: 'Putih'}

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)

# Initialize session state for uploaded files and predictions
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Navigation
menu = ["Overview", "Prediksi"]
choice = st.sidebar.selectbox("Navigasi", menu)

# Overview page
if choice == "Overview":
    st.title("Sistem Klasifikasi Warna Matos Fashion")
    st.markdown(
        """
        Matos Fashion menghadirkan inovasi klasifikasi otomatis warna pakaian
        menggunakan kecerdasan buatan. Sistem ini membantu:
        - Mempermudah pengelolaan inventaris.
        - Memberikan pengalaman belanja yang lebih baik kepada pelanggan.
        - Menyediakan analisis akurat untuk warna produk (Merah, Kuning, Biru, Hitam, Putih).

        Klik tombol di bawah ini untuk mencoba fitur prediksi warna pakaian.
        """
    )
    if st.button("Coba Prediksi Warna"):
        st.experimental_set_query_params(page="Prediksi")

# Prediction page
elif choice == "Prediksi":
    st.title("Prediksi Warna Pakaian")

    # File uploader
    uploaded_files = st.file_uploader(
        "Unggah gambar pakaian (Maksimal 10 gambar)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # Add new uploaded files to session state
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file)

    # Display uploaded files and predictions
    if st.session_state.uploaded_files:
        col1, col2 = st.columns(2)

        # Loop through uploaded files
        for idx, uploaded_file in enumerate(st.session_state.uploaded_files):
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption=f"Gambar: {uploaded_file.name}", use_container_width=True)

            # Preprocess and predict if not already predicted
            if len(st.session_state.predictions) <= idx:
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_label = np.argmax(predictions)
                accuracy = np.max(predictions) * 100

                # Store predictions in session state
                st.session_state.predictions.append((uploaded_file.name, label_map[predicted_label], accuracy))

            # Display prediction results
            with col2:
                result = st.session_state.predictions[idx]
                st.write(f"**Warna:** {result[1]}")
                st.write(f"**Akurasi:** {result[2]:.2f}%")

    # Clear uploaded files and predictions
    if st.button("Hapus Gambar"):
        st.session_state.uploaded_files = []  # Clear uploaded files
        st.session_state.predictions = []  # Clear predictions
        st.experimental_rerun()  # Reload the app
