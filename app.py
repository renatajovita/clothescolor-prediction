import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set page configuration
st.set_page_config(page_title="Klasifikasi Warna Matos Fashion", layout="wide")

# Google Drive model link and path
MODEL_URL = "https://drive.google.com/uc?id=1bpm2Gp_qVqsBIMHw-kRlzHX0e3QNie70"
MODEL_PATH = "color_model.h5"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model berhasil diunduh!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Map label indices to color names
label_map = {0: 'Merah', 1: 'Kuning', 2: 'Biru', 3: 'Hitam', 4: 'Putih'}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Initialize session states
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = []
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

# Navigation menu
menu = ["Overview", "Prediksi"]
choice = st.sidebar.selectbox("Navigasi", menu)

# Overview page
if choice == "Overview":
    st.title("Sistem Klasifikasi Warna Matos Fashion")
    st.markdown(
        """
        Sistem klasifikasi otomatis warna pakaian menggunakan kecerdasan buatan:
        - Mempermudah pengelolaan inventaris.
        - Memberikan pengalaman belanja yang lebih baik.
        - Menyediakan analisis akurat untuk warna produk: Merah, Kuning, Biru, Hitam, Putih.

        Klik tombol di bawah untuk mencoba fitur prediksi.
        """
    )

# Prediction page
elif choice == "Prediksi":
    st.title("Prediksi Warna Pakaian")

    # Handle reset if triggered
    if st.session_state.reset_triggered:
        st.session_state.uploaded_files = []
        st.session_state.results = []
        st.session_state.reset_triggered = False  # Reset the flag
        st.info("Semua gambar dan hasil prediksi telah dihapus. Silakan unggah gambar baru.")

    # File uploader for images
    uploaded_files = st.file_uploader(
        "Unggah gambar pakaian (Maksimal 10 gambar)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    # Save uploaded files to session state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.results = []  # Clear previous results

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)

            # Model prediction
            predictions = model.predict(processed_image)
            predicted_label = np.argmax(predictions)
            accuracy = np.max(predictions) * 100
            color_name = label_map[predicted_label]

            # Save results in session state
            st.session_state.results.append(
                {
                    "file_name": uploaded_file.name,
                    "color": color_name,
                    "accuracy": accuracy,
                    "image": image
                }
            )

    # Display predictions if results exist
    if st.session_state.results:
        st.markdown("### Hasil Prediksi")
        for result in st.session_state.results:
            st.image(result["image"], caption=f"Gambar: {result['file_name']}", use_container_width=True)
            st.write(f"**Warna:** {result['color']}")
            st.write(f"**Akurasi:** {result['accuracy']:.2f}%")

    # Reset button to clear uploaded files and predictions
if st.button("Hapus Gambar"):
    st.session_state.uploaded_files = []  # Reset file yang diunggah
    st.session_state.results = []  # Reset hasil prediksi
    st.session_state.clear_trigger = True  # Tandai bahwa reset telah dilakukan
    st.query_params(page="Prediksi")  # Refresh ke halaman Prediksi
    st.success("Semua gambar dan hasil prediksi telah dihapus. Silakan unggah gambar baru.")


