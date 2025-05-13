!pip install -U torchaudio --quiet

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ 오디오 모델 정의
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T, 128]
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

# ✅ 전처리
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

# ✅ 커스텀 데이터셋
class AudioDataset(Dataset):
    def __init__(self, df, mel_spec):
        self.df = df
        self.mel_spec = mel_spec

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['audio_path'])
        mel = self.mel_spec(waveform).squeeze(0).transpose(0, 1)  # [T, 128]
        label = torch.tensor(row['label'], dtype=torch.float)
        return mel, label

# ✅ 데이터 로딩
df = pd.read_csv("/content/train_audio.csv")  # audio_path, label
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_ds = AudioDataset(train_df, mel_spec)
val_ds = AudioDataset(val_df, mel_spec)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ✅ 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
best_f1 = 0.0

# ✅ 학습 루프
for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_audio_model.pth")
        print("✅ 오디오 모델 저장 완료")
