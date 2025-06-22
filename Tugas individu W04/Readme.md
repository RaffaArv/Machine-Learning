# 🗑️ Klasifikasi Sampah Organik vs Anorganik Menggunakan Transfer Learning (ResNet50)

Proyek ini merupakan implementasi model klasifikasi gambar dua kelas: sampah organik dan sampah anorganik, menggunakan metode transfer learning dengan pretrained model ResNet50.

---

## 📌 Deskripsi Proyek

Pengelolaan sampah menjadi salah satu tantangan lingkungan yang perlu ditangani secara efisien. Salah satu langkah awal adalah dengan memilah sampah organik dan anorganik. Model ini bertujuan untuk membantu klasifikasi otomatis gambar sampah berdasarkan jenisnya, dengan memanfaatkan kemampuan pretrained model ResNet50.

---

## 🧠 Tujuan

- Membangun model klasifikasi gambar dua kelas (organik vs anorganik).
- Menerapkan metode transfer learning untuk mengoptimalkan performa dengan dataset terbatas.
- Mengevaluasi hasil menggunakan metrik akurasi, confusion matrix, dan classification report.

---

## 📂 Dataset

Dataset terdiri dari:
- `organik/`: 100 gambar (misalnya: daun kering, sisa makanan)
- `anorganik/`: 100 gambar (misalnya: botol plastik, kaleng, kertas)

### Struktur Folder:
dataset/
├── organik/
│ ├── daun.jpg
│ └── ...
└── anorganik/
├── botol.jpg
└── ...


---

## ⚙️ Teknologi dan Library

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

---

## 🛠️ Implementasi Model

```python
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Data generator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/dataset-sampah-organik-anorganik/dataset-sampah-organik-anorganik',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    '/kaggle/input/dataset-sampah-organik-anorganik/dataset-sampah-organik-anorganik',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Transfer learning: ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


# Hasil 
              precision    recall  f1-score   support

    organik       0.90      0.95      0.92        20
  anorganik       0.94      0.90      0.92        20

    accuracy                           0.92        40
