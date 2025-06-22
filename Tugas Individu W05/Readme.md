# 📄 Tugas Individu: Klasifikasi Teks dengan RNN (LSTM)

Proyek ini merupakan implementasi klasifikasi teks menggunakan model Recurrent Neural Network (RNN), khususnya arsitektur Long Short-Term Memory (LSTM), untuk membedakan pesan Spam dan Non-Spam berdasarkan teks SMS.

---

## 📁 Struktur File

| Nama File | Deskripsi |
|-----------|-----------|
| `text_classification_rnn.ipynb` | Notebook Jupyter berisi semua langkah preprocessing, modeling, training, dan evaluasi |
| `Laporan_Tugas_RNN.docx` | Laporan lengkap tugas individu dalam format Word |
| `README.md` | Ringkasan proyek dan petunjuk penggunaan |

---

## 📊 Dataset

- Sumber: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Isi: 5572 pesan teks SMS yang telah dilabeli sebagai `spam` atau `ham` (non-spam)
- Format: TSV (Tab-separated values)
- Distribusi:
  - Spam: 747 pesan
  - Non-spam: 4825 pesan

---

## ⚙️ Cara Menjalankan

1. Buka file `text_classification_rnn.ipynb` di Google Colab atau Jupyter Notebook.
2. Jalankan sel secara berurutan dari atas ke bawah.
3. Pastikan koneksi internet aktif karena dataset akan diunduh dari URL publik.
4. Setelah training, hasil evaluasi dan grafik akan muncul.

---

## 🧠 Model & Konfigurasi

- Model: LSTM
- Arsitektur:
  - `Embedding(input_dim=5000, output_dim=64)`
  - `LSTM(units=64, dropout=0.5)`
  - `Dense(1, activation='sigmoid')`
- Optimisasi:
  - Epoch: 5
  - Batch size: 32
  - Loss: `binary_crossentropy`
  - Optimizer: `adam`

---

## 📈 Hasil Evaluasi

- Akurasi Validasi: ~98%
- Confusion Matrix: Sangat baik dalam memisahkan kelas
- Visualisasi: Tersedia learning curve (akurasi per epoch)

---

## ✨ Insight & Refleksi

- Preprocessing sangat penting: pembersihan teks dan padding berpengaruh besar.
- Dropout 0.5 membantu menghindari overfitting.
- LSTM bekerja sangat baik meskipun arsitektur sederhana.

---

## 📚 Referensi

- [UCI SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

---

> Tugas ini disusun untuk mata kuliah Pemrosesan Bahasa Alami sebagai bagian dari penilaian individu dalam membangun dan memahami model klasifikasi berbasis deep learning.

