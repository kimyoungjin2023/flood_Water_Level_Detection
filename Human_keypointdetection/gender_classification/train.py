import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ─────────────────────────────
# 설정
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224

DATA_DIR = "./datasets"   # ← 데이터셋 루트 (Training / Validation 포함)
MODEL_SAVE_PATH = "gender_classifier.pth"


# ─────────────────────────────
# 1. 모델
# ─────────────────────────────
class GenderClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        in_feat = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # female / male
        )

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────
# 2. 데이터 로더
# ─────────────────────────────
def get_dataloaders(data_dir):

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    train_path = os.path.join(data_dir, "Training")
    val_path   = os.path.join(data_dir, "Validation")

    train_ds = ImageFolder(train_path, transform=train_tf)
    val_ds   = ImageFolder(val_path,   transform=val_tf)

    print("[INFO] 클래스 인덱스:", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,        # 🔥 Windows 안정 설정
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"[INFO] Train 데이터: {len(train_ds)}장")
    print(f"[INFO] Val 데이터:   {len(val_ds)}장")

    return train_loader, val_loader


# ─────────────────────────────
# 3. 학습
# ─────────────────────────────
def train():

    train_loader, val_loader = get_dataloaders(DATA_DIR)

    model = GenderClassifier(pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):

        # ── Train ──
        model.train()
        total, correct = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # ── Validation ──
        model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                preds = model(imgs).argmax(1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100

        print(f"Epoch {epoch+1:02d} | Train {train_acc:.1f}% | Val {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ 모델 저장 (Val {val_acc:.1f}%)")

    print(f"\n✅ 완료! Best Val Acc: {best_acc:.1f}%")


# ─────────────────────────────
# 실행
# ─────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Device: {DEVICE}")
    train()