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

    # Tombol Reset untuk menghapus semua gambar dan prediksi
    if st.button("Reset"):
        st.session_state.uploaded_files = []  # Reset semua gambar
        st.session_state.results = []  # Reset semua hasil prediksi
        st.info("Semua gambar dan hasil prediksi telah direset. Silakan unggah gambar baru.")

    # File uploader untuk unggah gambar
    uploaded_files = st.file_uploader(
        "Unggah gambar pakaian (Maksimal 10 gambar)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    # Jika ada gambar yang diunggah
    if uploaded_files:
        # Tambahkan gambar baru ke session state
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file)

        # Proses setiap gambar yang ada di session state
        st.session_state.results = []  # Reset hasil prediksi setiap unggah baru
        for uploaded_file in st.session_state.uploaded_files:
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)

            # Prediksi dengan model
            predictions = model.predict(processed_image)
            predicted_label = np.argmax(predictions)
            accuracy = np.max(predictions) * 100
            color_name = label_map[predicted_label]

            # Simpan hasil prediksi ke session state
            st.session_state.results.append(
                {
                    "file_name": uploaded_file.name,
                    "color": color_name,
                    "accuracy": accuracy,
                    "image": image
                }
            )

    # Tampilkan hasil prediksi
    if st.session_state.results:
        st.markdown("### Hasil Prediksi")
        for idx, result in enumerate(st.session_state.results):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(result["image"], caption=f"Gambar: {result['file_name']}", use_container_width=True)
                st.write(f"**Warna:** {result['color']}")
                st.write(f"**Akurasi:** {result['accuracy']:.2f}%")
            with col2:
                # Tombol silang untuk menghapus gambar tertentu
                if st.button(f"Hapus {result['file_name']}", key=f"hapus_{idx}"):
                    # Hapus gambar dan hasil prediksi tertentu
                    st.session_state.uploaded_files.pop(idx)
                    st.session_state.results.pop(idx)
                    st.experimental_update()  # Update tampilan tanpa reload
