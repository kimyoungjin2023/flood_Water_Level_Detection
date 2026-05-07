import json
import os

def convert_highway_json(json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
   
    for file_name in os.listdir(json_folder):
        if not file_name.endswith('.json'):
            continue
           
        with open(os.path.join(json_folder, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
       
        # 이미지 정보 가져오기 (1920x1080 등)
        img_info = data['images'][0]
        img_w = img_info['width']
        img_h = img_info['height']
        img_id = img_info['file_name'].replace('.jpg', '')
       
        yolo_results = []
       
        for ann in data['annotations']:
            # 'bbox' 키가 없는 요소(labelingcount 등)는 건너뜀
            if 'bbox' not in ann:
                continue
               
            # 도로공사 특유의 구조: bbox = [[x, y], [w, h]]
            pos = ann['bbox'][0]
            size = ann['bbox'][1]
           
            x_min, y_min = pos[0], pos[1]
            w, h = size[0], size[1]
           
            # YOLO 포맷: 중심점 x, y 및 너비, 높이 (0~1 정규화)
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
           
            # category_id 사용 (제공된 JSON에선 car가 0번)
            class_id = ann['category_id']
           
            yolo_results.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
           
        # YOLO 텍스트 파일 저장
        with open(os.path.join(output_folder, f"{img_id}.txt"), 'w') as f:
            f.write('\n'.join(yolo_results))

# 경로 설정 후 실행
convert_highway_json('./datasets/Validation/LabelingData', './datasets/val/labels')