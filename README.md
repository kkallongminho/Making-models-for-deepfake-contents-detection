# 🎧🖼️ Multimodal Binary Classification (Image + Audio)

This project implements **two independent binary classifiers**:

- 🖼️ An **image classifier** using EfficientNet-B3
- 🎧 An **audio classifier** using a Bi-directional LSTM on mel-spectrograms

Each model is trained separately and predicts whether an input is **real (1)** or **fake (0)**. Best models are saved based on validation F1 score.

---

---

## 🧠 Model Architectures

### 1. 🖼️ Image Classifier (EfficientNet-B3)
- Backbone: `timm.create_model('efficientnet_b3')`
- Head: Global Avg Pool + FC Layer → Sigmoid
- Optimizer: `AdamW`
- Loss: `BCELoss`

### 2. 🎧 Audio Classifier (LSTM)
- Input: `torchaudio.MelSpectrogram` → `[T, 128]`
- Model: 2-layer BiLSTM + FC → Sigmoid
- Optimizer: `AdamW`
- Loss: `BCELoss`

---

## 📑 Dataset Format

### train_images.csv
```csv
image_path,label
/path/to/image1.jpg,1
/path/to/image2.jpg,0

## train_audio.csv
audio_path,label
/path/to/audio1.wav,1
/path/to/audio2.wav,0

##Output

Image
best_image_model.pth

Audio
best_audio_model.pth

