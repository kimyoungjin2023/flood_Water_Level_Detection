"""
child_classifier.py

파이프라인:
  사람 감지 (YOLOv8)
      │
      ▼
  어린이? ──YES──→ 스킵
      │
      NO
      ▼
  (다음 단계: 성별 분류 → Keypoint → Depth → 수위 cm)
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHILD_CUTOFF    = 12        # 12세 이하 = 어린이
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 100
LR              = 1e-4
MODEL_SAVE_PATH = "child_classifier_ver2.pth"

print(f"[INFO] Device: {DEVICE}")


# ─────────────────────────────────────────
# 1. 어린이 분류 모델 정의
# ─────────────────────────────────────────
class ChildClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        # 마지막 레이어 → 2클래스 (child / adult)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────
# 2. UTKFace 데이터셋
# ─────────────────────────────────────────
class UTKFaceDataset(Dataset):
    """
    UTKFace 파일명 형식: [age]_[gender]_[race]_[date].jpg
    label: 0 = 어린이 (0~12세), 1 = 성인 (13세+)

    transform을 외부에서 직접 넘기므로
    train/val 각각 다른 transform 적용 가능
    """
    def __init__(self, samples, transform=None):
        """
        samples  : [(경로, 라벨), ...] 리스트
        transform: torchvision transform
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_and_split(root_dir, val_ratio=0.2, seed=42):
    """
    UTKFace 폴더에서 샘플 로드 후 train/val 로 분리

    반환:
        train_samples : [(경로, 라벨), ...]  (80%)
        val_samples   : [(경로, 라벨), ...]  (20%)
    """
    import random

    all_samples = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        try:
            age   = int(fname.split("_")[0])
            label = 0 if age <= CHILD_CUTOFF else 1
            all_samples.append((os.path.join(root_dir, fname), label))
        except:
            continue

    # 재현성을 위해 시드 고정 후 셔플
    random.seed(seed)
    random.shuffle(all_samples)

    n       = len(all_samples)
    n_val   = int(n * val_ratio)
    n_train = n - n_val

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]

    # 분포 출력
    def stats(samples, name):
        nc = sum(1 for _, l in samples if l == 0)
        na = sum(1 for _, l in samples if l == 1)
        print(f"[{name}] 전체: {len(samples)}장 | 어린이: {nc}장 | 성인: {na}장")

    stats(train_samples, "TRAIN")
    stats(val_samples,   "VAL  ")

    return train_samples, val_samples


# ─────────────────────────────────────────
# 3. 학습
# ─────────────────────────────────────────
def train(data_dir="./UTKFace"):

    # ── Transform 정의 ──────────────────────
    # train: augmentation 포함
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # val: augmentation 없음 (정확한 평가를 위해)
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ── 데이터 로드 및 분할 ──────────────────
    train_samples, val_samples = load_and_split(data_dir, val_ratio=0.2)

    # train/val 각각 독립된 Dataset 생성 → transform 섞일 일 없음
    train_ds = UTKFaceDataset(train_samples, transform=train_tf)
    val_ds   = UTKFaceDataset(val_samples,   transform=val_tf)

    n_train = len(train_ds)
    n_val   = len(val_ds)

    # ── 클래스 불균형 처리 (성인이 훨씬 많음 → WeightedSampler) ──
    labels  = [l for _, l in train_samples]
    n_child = labels.count(0)
    n_adult = labels.count(1)
    w       = [1.0 / n_child if l == 0 else 1.0 / n_adult for l in labels]
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # 모델
    model     = ChildClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: backbone 동결, classifier만 학습
    for p in model.backbone.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # Phase 2: 10에폭 후 backbone 상위 3블록 해제
        if epoch == 10:
            print("\n[INFO] Phase 2 시작: backbone 상위 3블록 학습 활성화")
            for p in model.backbone.features[-3:].parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": model.backbone.features[-3:].parameters(),
                "lr":     LR * 0.1
            })

        # Train
        model.train()
        total_loss, correct = 0.0, 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]",
                                  leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()

        train_acc = correct / n_train * 100

        # Validation
        model.eval()
        val_correct = 0
        child_correct, child_total = 0, 0
        adult_correct, adult_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()

                # 클래스별 정확도
                cm = (labels == 0)
                am = (labels == 1)
                child_correct += (preds[cm] == 0).sum().item()
                adult_correct += (preds[am] == 1).sum().item()
                child_total   += cm.sum().item()
                adult_total   += am.sum().item()

        val_acc   = val_correct   / n_val              * 100
        child_acc = child_correct / max(child_total, 1) * 100
        adult_acc = adult_correct / max(adult_total, 1) * 100

        print(f"Epoch {epoch+1:02d} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"어린이: {child_acc:.1f}% | "
              f"성인: {adult_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ 모델 저장 (val_acc={val_acc:.1f}%)")

        scheduler.step()

    print(f"\n✅ 학습 완료! Best Val Accuracy: {best_val_acc:.1f}%")


# ─────────────────────────────────────────
# 4. 추론 클래스
# ─────────────────────────────────────────
class ChildClassifierInference:
    """
    사용법:
        clf = ChildClassifierInference("child_classifier.pth")
        result = clf.predict(bgr_crop)
        # result = {"is_child": True/False, "confidence": 0.95}
    """
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model = ChildClassifier(pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.model.eval().to(DEVICE)

        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, bgr_crop):
        """
        bgr_crop: OpenCV BGR 이미지 (사람 crop)
        반환: {"is_child": bool, "label": str, "confidence": float}
        """
        img    = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        tensor = self.tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        cls  = probs.argmax().item()   # 0=child, 1=adult
        conf = probs[cls].item()

        return {
            "is_child":   cls == 0,
            "label":      "child" if cls == 0 else "adult",
            "confidence": round(conf, 3),
            "child_prob": round(probs[0].item(), 3),
            "adult_prob": round(probs[1].item(), 3)
        }


# ─────────────────────────────────────────
# 5. 사람 감지 + 어린이 분류 연결
# ─────────────────────────────────────────
class PersonDetectorWithChildFilter:
    """
    사람 감지 (YOLOv8) → 어린이 필터링 → 성인만 반환

    사용법:
        detector = PersonDetectorWithChildFilter()
        adults   = detector.process(frame)
        # adults = [{"bbox": [x1,y1,x2,y2], "crop": img}, ...]
    """
    def __init__(self,
                 yolo_model="yolov8n.pt",
                 classifier_path=MODEL_SAVE_PATH,
                 conf_threshold=0.5):

        print("[INFO] YOLOv8 로드 중...")
        self.yolo       = YOLO(yolo_model)
        print("[INFO] 어린이 분류기 로드 중...")
        self.classifier = ChildClassifierInference(classifier_path)
        self.conf_th    = conf_threshold

    def process(self, frame):
        """
        frame: OpenCV BGR 이미지
        반환:
            adults   → 다음 단계(성별분류→Keypoint→Depth)로 전달
            skipped  → 어린이라서 제외된 목록 (로그용)
        """
        results = self.yolo(frame, classes=[0], verbose=False)  # class 0 = person

        adults  = []
        skipped = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.conf_th:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 너무 작은 박스 스킵 (노이즈)
            if (x2 - x1) < 30 or (y2 - y1) < 60:
                continue

            crop = frame[y1:y2, x1:x2]

            # 어린이 분류
            clf_result = self.classifier.predict(crop)

            person_info = {
                "bbox":       [x1, y1, x2, y2],
                "crop":       crop,
                "is_child":   clf_result["is_child"],
                "label":      clf_result["label"],
                "confidence": clf_result["confidence"]
            }

            if clf_result["is_child"]:
                skipped.append(person_info)   # 어린이 → 스킵
            else:
                adults.append(person_info)    # 성인 → 다음 단계로

        return adults, skipped


# ─────────────────────────────────────────
# 6. 실행 (테스트)
# ─────────────────────────────────────────
def run_video(video_path="0"):
    """
    video_path: 영상 파일 경로 or "0" (웹캠)
    """
    detector = PersonDetectorWithChildFilter()

    cap = cv2.VideoCapture(int(video_path) if video_path == "0" else video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        adults, skipped = detector.process(frame)

        # 시각화
        for p in adults:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ADULT {p['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for p in skipped:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"CHILD (skip) {p['confidence']:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 상태 출력
        print(f"성인: {len(adults)}명 | 어린이(스킵): {len(skipped)}명")

        cv2.imshow("Flood Detection - Person Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # 학습 모드
        # python child_classifier.py train ./UTKFace
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "./UTKFace"
        train(data_dir)

    else:
        # 실행 모드 (학습된 모델 필요)
        # python child_classifier.py
        # python child_classifier.py ./video.mp4
        video = sys.argv[1] if len(sys.argv) > 1 else "0"
        run_video(video)