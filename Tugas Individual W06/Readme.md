# 🧠 Eksplorasi Autoencoder dengan CNN pada Fashion MNIST

Proyek ini merupakan eksplorasi arsitektur autoencoder berbasis Convolutional Neural Network (CNN) untuk merekonstruksi gambar dari dataset **Fashion MNIST**. Fokus utama adalah memahami struktur encoder-decoder, menganalisis representasi laten, dan mengevaluasi kualitas rekonstruksi gambar.

---

## 🎯 Tujuan

- Membangun model autoencoder dengan minimal 3 layer encoder dan 3 layer decoder.
- Mengeksplorasi kualitas representasi laten dari gambar mode.
- Mengevaluasi kemampuan rekonstruksi berdasarkan variasi struktur model.

---

## 🧠 Arsitektur Model

Model yang dibangun adalah **Convolutional Autoencoder (CAE)** dengan arsitektur sebagai berikut:

### Encoder
- Conv2D(1 → 16), kernel 3, stride 2 → output: 14×14
- Conv2D(16 → 32), kernel 3, stride 2 → output: 7×7
- Conv2D(32 → 64), stride 1, padding 1 → output: 7×7
- Flatten → FC ke latent vector (32 dimensi)

### Decoder
- FC dari 32 → 64×7×7 → reshape
- ConvTranspose2D(64 → 32), kernel 3, stride 1 → output: 7×7
- ConvTranspose2D(32 → 16), stride 2 → 14×14
- ConvTranspose2D(16 → 1), stride 2 → 28×28
- Output activation: Sigmoid

Model ini menjaga ukuran input dan output tetap 28×28 agar loss dapat dihitung tanpa error dimensi.

---

## 🧪 Dataset

- Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Format: Grayscale 1 channel, ukuran 28×28 piksel
- Jumlah:
  - 60.000 gambar training
  - 10.000 gambar testing

---

## 📊 Hasil Eksperimen

### 📉 Loss Training

Model dilatih selama 20 epoch menggunakan MSELoss dan optimizer Adam. Loss menurun stabil selama proses training:
Epoch 1, Loss: 0.0367
Epoch 2, Loss: 0.0149
Epoch 3, Loss: 0.0121
Epoch 4, Loss: 0.0107
Epoch 5, Loss: 0.0099
Epoch 6, Loss: 0.0093
Epoch 7, Loss: 0.0090
Epoch 8, Loss: 0.0087
Epoch 9, Loss: 0.0085
Epoch 10, Loss: 0.0083
Epoch 11, Loss: 0.0082
Epoch 12, Loss: 0.0080
Epoch 13, Loss: 0.0080
Epoch 14, Loss: 0.0079
Epoch 15, Loss: 0.0078
Epoch 16, Loss: 0.0077
Epoch 17, Loss: 0.0076
Epoch 18, Loss: 0.0076
Epoch 19, Loss: 0.0075
Epoch 20, Loss: 0.0075

### 🔁 Rekonstruksi Gambar

Model berhasil merekonstruksi bentuk utama objek mode seperti sepatu, baju, dan celana dengan cukup baik. Walaupun detail tekstur tidak sempurna, struktur global dapat dikenali.

Contoh visualisasi input vs output:

| Input | Output |
|-------|--------|
| ![](results/input_1.png) | ![](results/output_1.png) |

> (Gambar hasil bisa disimpan di folder `results/`)

---

## 🚀 Cara Menjalankan

1. Clone atau buka notebook:
   - `autoencoder_fashionmnist.ipynb` atau `autoencoder_fashionmnist.py`
2. Install dependensi:
   ```bash
   pip install torch torchvision matplotlib
3. Jalankan
   python autoencoder_fashionmnist.py

📚 Referensi
- Fashion MNIST Dataset
- PyTorch Documentation
- Autoencoders - Deep Learning Book
