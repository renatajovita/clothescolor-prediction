import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Warna Pakaian", layout="wide")

# URL model dan path lokal
MODEL_URL = "https://drive.google.com/uc?id=1bpm2Gp_qVqsBIMHw-kRlzHX0e3QNie70"
MODEL_PATH = "color_model.h5"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model berhasil diunduh!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Label warna
label_map = {0: 'Merah', 1: 'Kuning', 2: 'Biru', 3: 'Hitam', 4: 'Putih'}

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalisasi
    return np.expand_dims(image, axis=0)

# Inisialisasi session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = []
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = False

# Navigasi
menu = ["Overview", "Prediksi"]
choice = st.sidebar.selectbox("Navigasi", menu)

# Halaman Overview
if choice == "Overview":
    st.title("Sistem Klasifikasi Warna Pakaian")
    st.markdown("""
    **Selamat datang di aplikasi prediksi warna pakaian!**
    
    - Unggah gambar pakaian Anda.
    - Sistem akan memberikan prediksi warna berdasarkan model AI.
    - Warna yang dapat diprediksi: Merah, Kuning, Biru, Hitam, Putih.
    """)

# Halaman Prediksi
elif choice == "Prediksi":
    st.title("Prediksi Warna Pakaian")

    # Tombol Hapus untuk reset
    if st.session_state.clear_trigger:
        st.session_state.uploaded_files = []
        st.session_state.results = []
        st.session_state.clear_trigger = False
        st.experimental_rerun()  # Reload halaman

    # File uploader
    uploaded_files = st.file_uploader(
        "Unggah gambar pakaian (Maksimal 10 gambar)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

        col1, col2 = st.columns(2)
        results = []

        # Prediksi untuk setiap gambar yang diunggah
        for uploaded_file in st.session_state.uploaded_files:
            try:
                image = Image.open(uploaded_file)
                with col1:
                    st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)

                # Preprocess dan prediksi
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_label = np.argmax(predictions)
                accuracy = np.max(predictions) * 100
                color_name = label_map[predicted_label]
                results.append((uploaded_file.name, color_name, accuracy))

                with col2:
                    st.write(f"**Warna:** {color_name}")
                    st.write(f"**Akurasi:** {accuracy:.2f}%")

            except Exception as e:
                st.error(f"Terjadi error pada file {uploaded_file.name}: {e}")

        st.session_state.results = results

    # Tombol Hapus
    if st.button("Hapus Gambar"):
        st.session_state.clear_trigger = True
        st.experimental_rerun()

    # Menampilkan hasil prediksi
    if st.session_state.results:
        st.markdown("### Hasil Prediksi")
        for result in st.session_state.results:
            st.write(f"File: {result[0]} | Warna: {result[1]} | Akurasi: {result[2]:.2f}%")
