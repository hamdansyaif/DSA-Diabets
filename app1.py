import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd # Import pandas untuk membuat DataFrame

# Tentukan path model
model_path = 'lgbm_model.pkl'

# Load model dengan penanganan error
model = None # Inisialisasi model sebagai None
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.error(f"Error: File model '{model_path}' tidak ditemukan. Pastikan file model berada di direktori yang sama dengan aplikasi.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

st.title("Prediksi Diabetes dengan Machine Learning")

# Buat tab
tab_prediksi, tab_tentang_model = st.tabs(["Prediksi Diabetes", "Tentang Model"])

with tab_prediksi:
    st.header("Input Data Pasien")

    # --- Tambahan Kode untuk Tabel Informasi BMI ---
    st.markdown("#### Informasi Klasifikasi BMI")
    # Definisikan data untuk tabel
    bmi_info_data = {
        "BMI Range": ["< 18.5", "18.5 - 24.9", "24.9 - 29.9", "29.9 - 34.9", "34.9 - 39.9", "39.9 ke atas"],
        "BMI Class": ["0 (Underweight)", "1 (Normal Weight)", "2 (Overweight)", "3 (Obesity I)", "4 (Obesity II)", "5 (Obesity III)"]
    }
    df_bmi_info = pd.DataFrame(bmi_info_data)
    st.table(df_bmi_info) # Menampilkan DataFrame sebagai tabel di Streamlit
    # --- Akhir Tambahan Kode ---

    # Bagian input data pasien dengan validasi sederhana
    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=30)
        hipertensi = st.selectbox("Hipertensi", ["Tidak", "Ya"], help="Apakah pasien memiliki riwayat hipertensi?")
        heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"], help="Apakah pasien memiliki riwayat penyakit jantung?")

    with col2:
        hba1c = st.number_input("Kadar HbA1c (%)", min_value=3.0, max_value=20.0, value=5.5, step=0.1, help="Hemoglobin glikosilasi")
        blood_glucose = st.number_input("Kadar Glukosa Darah (mg/dL)", min_value=50, max_value=500, value=100)
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=22.0, step=0.1, help="Body Mass Index")

    # Encode categorical variables
    hipertensi_val = 1 if hipertensi == "Ya" else 0
    heart_disease_val = 1 if heart_disease == "Ya" else 0

    def get_bmi_class(bmi_value):
        if bmi_value < 18.5:
            return 0 # Underweight
        elif 18.5 <= bmi_value < 24.9:
            return 1 # Normal weight
        elif 24.9 <= bmi_value < 29.9:
            return 2 # Overweight
        elif 29.9 <= bmi_value < 34.9:
            return 3 # Obesity I
        elif 34.9 <= bmi_value < 39.9:
            return 4 # Obesity II
        else: # This 'else' covers bmi_value >= 39.9
            return 5 # Obesity III

    bmi_class_val = get_bmi_class(bmi)

    # Tetapkan nilai threshold secara fix
    prediction_threshold = 0.5

    if st.button("Prediksi Diabetes"):
        # Validasi minimal
        if not all([umur, hba1c, blood_glucose, bmi]):
            st.warning("Mohon lengkapi semua data input.")
        else:
            # Menampilkan spinner saat prediksi berlangsung
            with st.spinner('Melakukan prediksi...'):
                try:
                    data = np.array([[
                        umur,
                        hipertensi_val,
                        heart_disease_val,
                        hba1c,
                        blood_glucose,
                        bmi_class_val
                    ]])

                    # Gunakan predict_proba() untuk mendapatkan probabilitas
                    probabilities = model.predict_proba(data)[0]
                    probability_diabetes = probabilities[1] # Probabilitas untuk kelas 1 (Diabetes POSITIF)

                    # Bandingkan probabilitas dengan threshold yang fix
                    pred_class = 1 if probability_diabetes >= prediction_threshold else 0

                    hasil = "POSITIF Diabetes" if pred_class == 1 else "NEGATIF Diabetes"

                    # Tampilkan probabilitas dan threshold yang digunakan
                    st.write(f"Probabilitas Diabetes: **{probability_diabetes:.2f}**")
                    st.write(f"Ambang Batas Prediksi Digunakan (Fix): **{prediction_threshold:.2f}**")

                    if hasil == "POSITIF Diabetes":
                        st.error(f"### Hasil Prediksi: {hasil}")
                        st.markdown("⚠️ Disarankan untuk berkonsultasi dengan dokter untuk diagnosis lebih lanjut.")
                    else:
                        st.success(f"### Hasil Prediksi: {hasil}")
                        st.markdown("✅ Berdasarkan data yang dimasukkan, risiko diabetes rendah. Tetap jaga pola hidup sehat!")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memprediksi: {e}")


with tab_tentang_model:
    st.header("Tentang Model LightGBM")

    st.write("Model machine learning yang digunakan adalah **LightGBM**.")
    st.write("") # Baris kosong untuk spasi
    st.write("**Mengapa LightGBM?**")
    st.write("Model ini dipilih berdasarkan metrik evaluasi yang kuat pada data uji:")
    st.write("") # Baris kosong untuk spasi

    st.subheader("Confusion Matrix:")
    # Data untuk Confusion Matrix
    confusion_matrix_data = pd.DataFrame(
        [[5630, 78],
         [200, 505]],
        columns=['Predicted Negatif', 'Predicted Positif'],
        index=['Actual Negatif', 'Actual Positif']
    )
    st.dataframe(confusion_matrix_data) # Menggunakan st.dataframe untuk tabel interaktif

    st.write("- True Negatives (TN): 5630 (Pasien tidak diabetes, diprediksi tidak diabetes)")
    st.write("- False Positives (FP): 78 (Pasien tidak diabetes, diprediksi diabetes)")
    st.write("- False Negatives (FN): 200 (Pasien diabetes, diprediksi tidak diabetes)")
    st.write("- True Positives (TP): 505 (Pasien diabetes, diprediksi diabetes)")
    st.write("") # Baris kosong untuk spasi

    st.subheader("Classification Report:")
    # Data untuk Classification Report
    classification_report_data = pd.DataFrame({
        'class': ['0', '1', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': [0.97, 0.72, '', 0.92, 0.95],
        'recall': [0.99, 0.78, '', 0.85, 0.96],
        'f1-score': [0.98, 0.78, 0.96, 0.88, 0.95],
        'support': [5708, 705, 6413, 6413, 6413]
    }).set_index('class')
    st.dataframe(classification_report_data) # Menggunakan st.dataframe untuk tabel interaktif

    st.write("**Penjelasan Metrik:**")
    st.write("* **Precision (Class 1 - Diabetes): 0.72**")
    st.write("  Dari semua yang diprediksi positif diabetes, 72% benar-benar diabetes.")
    st.write("* **Recall (Class 1 - Diabetes): 0.78**")
    st.write("  Dari semua pasien diabetes yang sebenarnya, model berhasil mendeteksi 78%.")
    st.write("* **F1-Score (Class 1 - Diabetes): 0.78**")
    st.write("  Rata-rata harmonik dari precision dan recall, menunjukkan keseimbangan.")
    st.write("* **Accuracy: 0.96**")
    st.write("  Overall akurasi model dalam memprediksi dengan benar.")
    st.write("") # Baris kosong untuk spasi

    st.subheader("LightGBM AUC pada Data Test: 0.9750")
    st.write("Kurva ROC (Receiver Operating Characteristic) menunjukkan kemampuan model untuk membedakan antara kelas positif dan negatif pada berbagai threshold. Nilai AUC (Area Under the Curve) sebesar **0.9750** menunjukkan kinerja diskriminasi model yang sangat baik. Semakin dekat ke 1, semakin baik.")
    # Anda bisa menambahkan gambar ROC Curve di sini jika file gambar tersedia
    st.image("image.png", caption="ROC Curve for LightGBM")


