import pandas as pd
import cv2
import os

image_dir = './datasets/data/training_images/'
label_dir = './datasets/data/labels/train/'
os.makedirs(label_dir, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv('./datasets/data/train_solution_bounding_boxes (1).csv')

for index, row in df.iterrows():
    img_name = row['image']
    img_path = os.path.join(image_dir, img_name)
   
    # 이미지 해상도 추출
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
   
    # YOLO 포맷으로 정규화 연산
    x_center = ((row['xmin'] + row['xmax']) / 2) / w
    y_center = ((row['ymin'] + row['ymax']) / 2) / h
    width = (row['xmax'] - row['xmin']) / w
    height = (row['ymax'] - row['ymin']) / h
   
    # 차량 클래스(0)로 txt 파일 저장
    txt_name = img_name.replace('.jpg', '.txt')
    with open(os.path.join(label_dir, txt_name), 'a') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")