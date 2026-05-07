import pandas as pd
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split

# 1. 기본 경로 설정
base_dir = './datasets/data/'
img_source_dir = os.path.join(base_dir, 'training_images')
csv_path = os.path.join(base_dir, 'train_solution_bounding_boxes (1).csv')

# 2. YOLO 정석 디렉토리 구조 생성
dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

# 3. 데이터 로드 및 분할 (데이터 누수 방지를 위해 이미지명 기준으로 분할)
df = pd.read_csv(csv_path)
unique_images = df['image'].unique()

# 8:2 비율로 Train / Val 분할
train_imgs, val_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)

def process_and_split(image_list, split_type):
    subset_df = df[df['image'].isin(image_list)]
   
    for img_name in image_list:
        src_img_path = os.path.join(img_source_dir, img_name)
        dst_img_path = os.path.join(base_dir, f'images/{split_type}', img_name)
       
        # 이미지 파일 복사
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            continue
           
        # 이미지 해상도 읽기
        img = cv2.imread(src_img_path)
        if img is None: continue
        h, w, _ = img.shape
       
        # 해당 이미지에 포함된 모든 차량의 바운딩 박스 추출
        boxes = subset_df[subset_df['image'] == img_name]
       
        txt_name = img_name.replace('.jpg', '.txt')
        txt_path = os.path.join(base_dir, f'labels/{split_type}', txt_name)
       
        # YOLO 정규화 포맷으로 저장
        with open(txt_path, 'w') as f:
            for _, row in boxes.iterrows():
                x_center = ((row['xmin'] + row['xmax']) / 2) / w
                y_center = ((row['ymin'] + row['ymax']) / 2) / h
                width = (row['xmax'] - row['xmin']) / w
                height = (row['ymax'] - row['ymin']) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 4. 함수 실행
print("Train 데이터 처리 중...")
process_and_split(train_imgs, 'train')

print("Val 데이터 처리 중...")
process_and_split(val_imgs, 'val')

print("데이터셋 분할 및 YOLO 포맷 변환 완료!")