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

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = []

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
        st.session_state.uploaded_files = uploaded_files
        st.session_state.results = []  # Reset previous results

        # Process each uploaded file
        for uploaded_file in st.session_state.uploaded_files:
            image = Image.open(uploaded_file)

            # Preprocess image
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_label = np.argmax(predictions)
            accuracy = np.max(predictions) * 100
            color_name = label_map[predicted_label]

            # Append result to session state
            st.session_state.results.append(
                {
                    "file_name": uploaded_file.name,
                    "color": color_name,
                    "accuracy": accuracy,
                    "image": image
                }
            )

    # Display results
    if st.session_state.results:
        st.markdown("### Hasil Prediksi")
        for result in st.session_state.results:
            st.image(result["image"], caption=f"Gambar: {result['file_name']}", use_container_width=True)
            st.write(f"**Warna:** {result['color']}")
            st.write(f"**Akurasi:** {result['accuracy']:.2f}%")

    # Clear uploaded files and results
    if st.button("Hapus Gambar"):
        st.session_state.uploaded_files.clear()  # Clear the list of uploaded files
        st.session_state.results.clear()  # Clear the results
        st.experimental_rerun()  # Reload the app
