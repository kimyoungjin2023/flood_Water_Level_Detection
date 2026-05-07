from pathlib import Path
from collections import Counter
import yaml

# =========================================================
# YOLO Dataset 자동 클래스 분석 + data.yaml 생성
# =========================================================

# 데이터셋 경로 (본인의 실제 경로로 수정하세요)
DATASET_PATH = r"./dataset"

# train / val 이미지 경로
TRAIN_PATH = "train/labels"
VAL_PATH = "val/labels"

# =========================================================
# label txt 전체 탐색
# =========================================================

label_files = list(Path(DATASET_PATH).rglob("*.txt"))

# classes.txt 제외
label_files = [
    f for f in label_files
    if f.name != "classes.txt"
]

print(f"\n총 Label 파일 수: {len(label_files)}")

# =========================================================
# 클래스 ID 수집
# =========================================================

class_counter = Counter()

for label_file in label_files:
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # YOLO format: 첫 번째 값이 class_id
            class_id = int(float(parts[0]))
            class_counter[class_id] += 1

    except Exception as e:
        print(f"오류 발생: {label_file}")
        print(e)

# =========================================================
# 클래스 정렬
# =========================================================

class_ids = sorted(class_counter.keys())

print("\n발견된 클래스 ID:")
print(class_ids)

# =========================================================
# 클래스 이름 자동 생성
# =========================================================
# 도로공사 데이터 가이드에 맞춰 추후 이름을 직접 수정해주면 더 좋습니다.
names_dict = {}
for class_id in class_ids:
    names_dict[class_id] = f"class_{class_id}"

# =========================================================
# data.yaml 생성
# =========================================================

data_yaml = {
    "path": DATASET_PATH,
    "train": TRAIN_PATH,
    "val": VAL_PATH,
    "nc": len(class_ids),
    "names": names_dict
}

yaml_save_path = Path(DATASET_PATH) / "data.yaml"

with open(yaml_save_path, "w", encoding="utf-8") as f:
    yaml.dump(
        data_yaml,
        f,
        allow_unicode=True,
        sort_keys=False
    )

# =========================================================
# 결과 출력
# =========================================================

print("\n================ 결과 ================\n")
print(f"총 클래스 개수: {len(class_ids)}\n")

for class_id in class_ids:
    print(
        f"Class ID: {class_id} "
        f"| 객체 수: {class_counter[class_id]}"
    )

print(f"\n[data.yaml 저장 완료]")
print(yaml_save_path.absolute()) # 절대 경로로 출력하면 확인이 더 편합니다.
print("\n======================================")