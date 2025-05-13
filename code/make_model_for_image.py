!pip install -U timm torchvision --quiet

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ✅ 이미지 모델 정의
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model("efficientnet_b3", pretrained=True, num_classes=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 커스텀 데이터셋
class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(Image.open(row['image_path']).convert("RGB"))
        label = torch.tensor(row['label'], dtype=torch.float)
        return image, label

# ✅ 데이터 로딩
df = pd.read_csv("/content/train_images.csv")  # image_path, label
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_ds = ImageDataset(train_df, transform)
val_ds = ImageDataset(val_df, transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ✅ 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)
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
        torch.save(model.state_dict(), "best_image_model.pth")
        print("✅ 이미지 모델 저장 완료")
