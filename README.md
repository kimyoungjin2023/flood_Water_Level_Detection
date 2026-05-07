# 지하공간 수위감지 프로젝트

- 침수 구역 분할(Sementic Segmentation)
- 기준객체에 대한 탐지
- 대략적 수위감지(먄약 기준객체가 미탐지경우(Depth Anything V2, DPT 모델 사용))

## 침수 구역 분할(Sementic Segmentation)

```
    U-Net 계열 모델 사용
            ▲
 위 모델들을 사용하여 test 진행
```

## 기준객체에 대한 탐지(Detection & Segmentation & Keypoint_Detection)

### 1. 기준 객체 선정
문헌 조사를 통해 대한민국 도로교통법 및 등등의 자료 참고

### 2. 기준 객체 탐지
아직까지의 4개의 기준객체 존재
1. Car_Part_Segmentation(차(바퀴, 사이드미러 등등))
2. Human_KeypointDetection(사람)
3. Sign_detection(표지판)
4. Tubular_Marker_detection(시선유도봉)

### 3. 기준객체 탐지 및 지정
자세한 내용은 모델 학습 폴더에 설명

## 기준 객체를 활용한 침수심 계산

### 기준객체 존재 O
- 기준객체가 존재하면 기하학적인 깊이 계산을 사용(정확한)
- Ex : 차량 바퀴 60cm 인데 50%가 잠긴경우 대략적인 깊이가 30cm로 확인 가능

### 기준객체 존재 X
- 기준객체가 존재하지 않으면 상대적인 깊이를 사용하여 깊이 계산(대략적인)
- 사용 모델 : Depth Anything V2, DPT, MiDaS 모델 사용 후 Test 진행

## 문제점

객체 탐지에 있어서 지하공간이라는 문제점 발생(어두운 상황)
실제 전처리 및 후처리를 사용하여 처리
실험 및 Test를 진행하면서 나오는 문제점 기입 및 해경방안 도출

## 오탐지 제거 방법 생각
- 오탐이미지에 빈 txt 라벨 추가