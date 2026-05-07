# 사람의 키를 이용한 수위감지

## 사람감지 후 성별 구분(classification) -> 대한민국 평균 남자 키 174cm, 평균 여자 키 160.5cm를 대입 후 Keypoint Detection(Human_Pose_Estimation)사용

문제점
어린이 감지 - 어린이는 키가 작음 classification에 추가를 할까 생각 중

### 원래 생각
```
사람 감지(YOLO11n)
    │
    ▼
성별 분류 → 남(174cm) / 여(160.5cm) 기준키 설정
    │
    ▼
Keypoint로 전신 픽셀 높이 계산(YOLO11n-Pose)
(머리 top ~ 발목 bottom)
    │
    ▼
픽셀/cm 비율 계산
    │
    ▼
수면선 픽셀 위치 감지
    │
    ▼
실제 수위 cm 출력!
```

### 변경된 생각
```
사람 감지(YOLO11n)
    │
    ▼
어린이 분류(어린이 사용 X)(EfficientNet-B0)
    │
    ▼
성별 분류 → 남(174cm) / 여(160.5cm) 기준키 설정(EfficientNet-B0)
    │
    ▼
Keypoint로 전신 픽셀 높이 계산(YOLO11n-Pose)
(머리 top ~ 발목 bottom)
    │
    ▼
픽셀/cm 비율 계산
    │
    ▼
수면선 픽셀 위치 감지
    │
    ▼
실제 수위 cm 출력!
```
---

학습 순서
 - 1번 어린이 구분(classification)
 - 2번 성별 구분(classification)
 - 3번 키 측정(Keypoint Detection)
 - Test 수위감지

---

이 파일에서 코드 확인 후 성능 이 좋으면 본 코드 적용
카메라 캘리브레이션 등등 본 코드에 적용
학습 전처리는 진행

---

ver2 사용하면 됨