# 🧠 Multimodal Binary Classifier (Image + Audio)

This project contains **training code only** for two deep learning models:

- 🖼️ `make_model_for_image.py`: trains an EfficientNet-B3 based image classifier
- 🎧 `make_model_for_Audio.py`: trains a BiLSTM-based audio classifier

> 📌 NOTE: This repository does **not include any pre-trained model files (`.pth`) or datasets**.  
You must prepare your own data folders (`images/`, `audios/`) before running the training scripts.

---

## 📁 Folder Structure (expected)
project/
├── images/                       # Folder with .jpg/.png image files
│   ├── real_001.jpg
│   ├── fake_002.jpg
│   └── …
├── audios/                       # Folder with .wav audio files
│   ├── real_001.wav
│   ├── fake_002.wav
│   └── …
├── make_model_for_image.py      # 🔧 Image classifier training script
├── make_model_for_Audio.py      # 🔧 Audio classifier training script
├── README.md                    # 📄 This file


---

## 🧠 What These Scripts Do

### 🖼️ `make_model_for_image.py`
- Model Name: **EfficientNet-B3**
- Framework: `timm`
- Loads all `.jpg`, `.png` files from `images/` folder
- Uses filenames to infer labels: files containing `"real"` → label `1`, else `0`
- Trains using `BCELoss` + `AdamW`
- Saves best model as `best_image_model.pth` (not included in this repo)

### 🎧 `make_model_for_Audio.py`
- Model Name: **2-layer BiLSTM + Fully Connected Head**
- Input: Mel-spectrograms (128 mel bins)
- Loads all `.wav` files from `audios/` folder
- Infers labels: `"real"` in filename → label `1`, else `0`
- Trains using `BCELoss` + `AdamW`
- Saves best model as `best_audio_model.pth` (not included)

---

## 🔧 Setup

pip install torch torchvision torchaudio timm scikit-learn pandas


🚀 Training

Ensure your images and audio files are named properly and placed in the correct folders.

Train Image Model
python make_model_for_image.py

Train Audio Model
python make_model_for_Audio.py

After training, you will find:
	•	best_image_model.pth
	•	best_audio_model.pth

These will be created only after successful training.
They are not included by default.


❗ Important
	•	No .pth models are included in this project.
	•	You must provide your own data and train the models from scratch using the scripts provided.

📌 License

MIT License — use for research and education.
