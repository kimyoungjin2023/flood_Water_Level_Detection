# 어린이 classification 모델 학습 스크립트
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, cv2, numpy as np
from tqdm import tqdm

# ─────────────────────────────
# 클래스 / 기준키
# ─────────────────────────────
AGE_CLASSES  = {0: "child", 1: "adult"}
CHILD_CUTOFF = 12  # 12세 이하 = 어린이

HEIGHT_REF = {
    "child": {"male": 128.0, "female": 127.0, "unknown": 127.5},
    "adult": {"male": 174.0, "female": 160.5, "unknown": 167.0}
}

# ─────────────────────────────
# 모델
# ─────────────────────────────
class ChildClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        # 마지막 레이어 2클래스로 교체
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)   # child / adult
        )
   
    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────
# 데이터셋 (UTKFace)
# ─────────────────────────────
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []
       
        for fname in os.listdir(root_dir):
            if not fname.lower().endswith('.jpg'):
                continue
            try:
                age   = int(fname.split('_')[0])
                label = 0 if age <= CHILD_CUTOFF else 1
                self.samples.append((
                    os.path.join(root_dir, fname), label
                ))
            except:
                continue
       
        # 클래스 분포 확인
        n_child = sum(1 for _, l in self.samples if l == 0)
        n_adult = sum(1 for _, l in self.samples if l == 1)
        print(f"데이터 로드 완료: 어린이 {n_child}장 | 성인 {n_adult}장")
   
    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────
# 학습
# ─────────────────────────────
def train(data_dir="./UTKFace", epochs=20, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
   
    # Transform
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
   
    # 데이터 분할
    full_ds = UTKFaceDataset(data_dir, transform=train_tf)
    n       = len(full_ds)
    n_train = int(n * 0.8)
    n_val   = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf
   
    # 클래스 불균형 처리 (성인이 훨씬 많음)
    labels     = [full_ds.samples[i][1] for i in train_ds.indices]
    n_child    = labels.count(0)
    n_adult    = labels.count(1)
    weights    = [n/n_child if l==0 else n/n_adult for l, n
                  in zip(labels, [n]*len(labels))]
    sampler    = torch.utils.data.WeightedRandomSampler(
        weights, len(weights)
    )
   
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)
   
    # 모델
    model     = ChildClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
   
    # Phase 1: classifier만 학습 (백본 동결)
    for p in model.backbone.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
   
    best_acc = 0.0
   
    for epoch in range(epochs):
        # Phase 2: 10에폭 후 백본 상위 레이어 해제
        if epoch == 10:
            print("\n>> Phase 2: 백본 상위 3블록 학습 시작")
            for p in model.backbone.features[-3:].parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                'params': model.backbone.features[-3:].parameters(),
                'lr':     lr * 0.1
            })
       
        # Train
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch+1}/{epochs}",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (model(imgs).argmax(1) == labels).sum().item()
       
        # Validation
        model.eval()
        val_correct, val_child_correct, val_adult_correct = 0, 0, 0
        val_child_total, val_adult_total = 0, 0
       
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
               
                # 클래스별 정확도
                child_mask = (labels == 0)
                adult_mask = (labels == 1)
                val_child_correct += (preds[child_mask] == 0).sum().item()
                val_adult_correct += (preds[adult_mask] == 1).sum().item()
                val_child_total   += child_mask.sum().item()
                val_adult_total   += adult_mask.sum().item()
       
        val_acc        = val_correct / n_val * 100
        child_acc      = val_child_correct / max(val_child_total,1) * 100
        adult_acc      = val_adult_correct / max(val_adult_total,1) * 100
       
        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"전체: {val_acc:.1f}% | "
              f"어린이: {child_acc:.1f}% | "
              f"성인: {adult_acc:.1f}%")
       
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'child_classifier.pth')
            print(f"  ✓ 저장 완료 (val_acc={val_acc:.1f}%)")
       
        scheduler.step()
   
    print(f"\n✅ 학습 완료! Best Accuracy: {best_acc:.1f}%")


# ─────────────────────────────
# 추론
# ─────────────────────────────
class ChildClassifierInference:
    def __init__(self, model_path='child_classifier.pth'):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = ChildClassifier(pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval().to(self.device)
       
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
   
    def predict(self, bgr_crop, gender="unknown"):
        img    = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        tensor = self.tf(img).unsqueeze(0).to(self.device)
       
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
       
        cls  = probs.argmax().item()
        conf = probs[cls].item()
        label = AGE_CLASSES[cls]          # "child" or "adult"
        ref_h = HEIGHT_REF[label][gender] # 기준키(cm)
       
        return {
            "label":         label,       # "child" / "adult"
            "confidence":    round(conf, 3),
            "ref_height_cm": ref_h,
            "child_prob":    round(probs[0].item(), 3),
            "adult_prob":    round(probs[1].item(), 3)
        }


if __name__ == "__main__":
    train(data_dir="./UTKFace", epochs=20)