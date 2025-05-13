!pip install -U torchaudio --quiet

import os
import glob
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ✅ 오디오 폴더에서 DataFrame 생성 (파일명에 real/fake 포함 시 라벨 자동 추정)
def get_audio_df_from_folder(folder_path):
    audio_paths = glob.glob(os.path.join(folder_path, "*.wav"))
    data = [{"audio_path": path, "label": 1 if "real" in os.path.basename(path).lower() else 0}
            for path in audio_paths]
    return pd.DataFrame(data)

# ✅ 설정
audio_folder = "/content/audios"  # 여기에 .wav 파일이 있어야 함
df = get_audio_df_from_folder(audio_folder)
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# ✅ Mel-Spectrogram 변환기
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

# ✅ 커스텀 Dataset 정의
class AudioDataset(Dataset):
    def __init__(self, df, mel_spec):
        self.df = df.reset_index(drop=True)
        self.mel_spec = mel_spec

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['audio_path'])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        mel = self.mel_spec(waveform).squeeze(0).transpose(0, 1)  # [T, 128]
        label = torch.tensor(row['label'], dtype=torch.float)
        return mel, label

# ✅ DataLoader 구성
train_ds = AudioDataset(train_df, mel_spec)
val_ds = AudioDataset(val_df, mel_spec)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ✅ 오디오 분류 모델 정의 (BiLSTM + FC)
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T, 128]
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])  # 마지막 timestep의 출력만 사용

# ✅ 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
best_f1 = 0.0

# ✅ 학습 루프
for epoch in range(3):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    # ✅ 검증
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            probs = model(x).squeeze()
            preds += (probs > 0.5).cpu().int().tolist()
            trues += y.tolist()
    f1 = f1_score(trues, preds)
    print(f"[Epoch {epoch+1}] Val F1: {f1:.4f}")

    # ✅ 최고 모델 저장
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_audio_model.pth")
        print("✅ best_audio_model.pth 저장 완료")
