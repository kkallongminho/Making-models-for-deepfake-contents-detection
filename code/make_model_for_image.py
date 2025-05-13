!pip install -U timm torchvision --quiet

import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ✅ 이미지 폴더에서 DataFrame 생성 (예: 파일명에 real/fake 포함 시)
def get_image_df_from_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.jpeg"))
    
    # 파일명 기준 라벨링: "real" 포함 시 1, 아니면 0
    data = [{"image_path": path, "label": 1 if "real" in os.path.basename(path).lower() else 0}
            for path in image_paths]
    return pd.DataFrame(data)

# ✅ 설정
image_folder = "/content/images"  # 실제 이미지가 저장된 경로
df = get_image_df_from_folder(image_folder)
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

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
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(Image.open(row['image_path']).convert("RGB"))
        label = torch.tensor(row['label'], dtype=torch.float)
        return image, label

# ✅ DataLoader
train_ds = ImageDataset(train_df, transform)
val_ds = ImageDataset(val_df, transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# ✅ EfficientNet-B3 모델 정의
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

# ✅ 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)
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

    # ✅ 모델 저장
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_image_model.pth")
        print("✅ best_image_model.pth 저장 완료")
