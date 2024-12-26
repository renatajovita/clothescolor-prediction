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

# Initialize session states for uploaded files and results
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = []

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

    # Check if reset state exists
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "results" not in st.session_state:
        st.session_state.results = []

    # Opsi untuk memilih sumber input gambar
    input_choice = st.radio(
        "Pilih sumber gambar:", 
        ("Unggah Gambar", "Gunakan Kamera")
    )

    if input_choice == "Unggah Gambar":
        # File uploader for images
        uploaded_files = st.file_uploader(
            "Unggah gambar pakaian (Maksimal 10 gambar)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

        # Detect when files are removed (cross-button click)
        if uploaded_files is None or len(uploaded_files) != len(st.session_state.uploaded_files):
            st.session_state.uploaded_files = []  # Clear uploaded files
            st.session_state.results = []  # Clear predictions

        # Save uploaded files to session state
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

    elif input_choice == "Gunakan Kamera":
        # Camera input
        camera_image = st.camera_input("Ambil gambar menggunakan kamera")

        if camera_image:
            # Save the captured image to session state
            st.session_state.uploaded_files = [camera_image]

    # Process images if available
    if st.session_state.uploaded_files:
        st.session_state.results = []  # Reset results for new uploads
        for uploaded_file in st.session_state.uploaded_files:
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
                    "file_name": "Kamera" if input_choice == "Gunakan Kamera" else uploaded_file.name,
                    "color": color_name,
                    "accuracy": accuracy,
                    "image": image
                }
            )

# Prediction page
elif choice == "Prediksi":
    st.title("Prediksi Warna Pakaian")

    # Check if reset state exists
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "results" not in st.session_state:
        st.session_state.results = []

    # Opsi untuk memilih sumber input gambar
    input_choice = st.radio(
        "Pilih sumber gambar:", 
        ("Unggah Gambar", "Gunakan Kamera")
    )

    if input_choice == "Unggah Gambar":
        # File uploader for images
        uploaded_files = st.file_uploader(
            "Unggah gambar pakaian (Maksimal 10 gambar)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

        # Update session state when new files are uploaded
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

    elif input_choice == "Gunakan Kamera":
        # Camera input
        camera_image = st.camera_input("Ambil gambar menggunakan kamera")

        if camera_image:
            # Save the captured image to session state
            st.session_state.uploaded_files = [camera_image]

    # Process images if available
    if st.session_state.uploaded_files:
        st.session_state.results = []  # Reset results for new uploads
        for uploaded_file in st.session_state.uploaded_files:
            # Open the image
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
                    "file_name": "Kamera" if input_choice == "Gunakan Kamera" else uploaded_file.name,
                    "color": color_name,
                    "accuracy": accuracy,
                    "image": image
                }
            )

    # Display predictions if results exist
    if st.session_state.results:
        st.markdown("### Hasil Prediksi")
        for i, result in enumerate(st.session_state.results):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.image(result["image"], caption=f"Gambar: {result['file_name']}", use_container_width=True)
                st.write(f"**Warna:** {result['color']}")
                st.write(f"**Akurasi:** {result['accuracy']:.2f}%")
            with col2:
                # Add delete button for each prediction
                if st.button(f"Hapus {result['file_name']}", key=f"delete_{i}"):
                    # Remove the specific result and file
                    st.session_state.results.pop(i)
                    st.session_state.uploaded_files.pop(i)
