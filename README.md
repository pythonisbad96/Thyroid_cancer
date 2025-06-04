# 🧠 갑상선암 진단 분류 AI 프로젝트

> 의료 데이터를 기반으로 갑상선암 여부를 예측하는 머신러닝 프로젝트입니다.  
> 다양한 모델 학습과 하이퍼파라미터 튜닝, 앙상블을 통해 F1-score를 극대화하였습니다.

---

## 📌 프로젝트 개요

- **목표**: 환자의 다양한 의학적 지표(TSH, T3, T4 등)를 바탕으로 갑상선암 여부(`Cancer`)를 분류하는 모델 개발
- **접근 방법**: 데이터 전처리 → 모델별 튜닝 → 앙상블 → Threshold 조정 → 성능 시각화
- **사용 기술**: Python, Scikit-learn, XGBoost, CatBoost, LightGBM, Matplotlib

---

## 🧾 데이터 정보

- **Train.csv**: 학습 및 검증 데이터
- **Test.csv**: 예측용 테스트 데이터
- **Sample_submission.csv**: 제출 형식 템플릿

| 컬럼명            | 설명                        |
|------------------|-----------------------------|
| TSH_Result, T3... | 갑상선 관련 호르몬 수치     |
| Nodule_Size      | 결절 크기 정보               |
| Family_Background| 가족력 여부                  |
| Cancer           | 타겟 변수 (0: 정상, 1: 암)  |

---

## 🔧 전처리 및 특성 선택

- Label Encoding을 통해 범주형 변수 처리
- `Gender`, `Smoke`, `Weight_Risk`, `Diabetes`는 feature 중요도가 낮아 제거
- 학습/검증 세트를 80:20 비율로 분리 (`stratify` 적용)

---

## 🤖 모델링 및 앙상블

### 🎯 XGBoost
- RandomizedSearchCV로 튜닝 (60회 반복)
- `scale_pos_weight`로 클래스 불균형 보정

### 🎯 CatBoost
- `auto_class_weights='Balanced'` 설정
- depth, learning_rate, iterations 튜닝

### 🎯 LightGBM
- 기본 모델 사용 + 클래스 가중치 적용

### 🧩 Soft Voting 앙상블
- 위 3가지 모델의 예측 확률을 평균 내어 최종 예측

---

## 🎯 Threshold 최적화

- 다양한 threshold(0.30 ~ 0.70)를 적용하여
- Validation set에서 **F1-score가 가장 높은 임계값**을 최종 threshold로 설정

---

## 📈 결과 및 평가

```text
- 최종 F1-score (검증 세트 기준): 약 0.93
- Precision, Recall, Confusion Matrix 등 포함
- ROC Curve 및 F1-score 변화 그래프 시각화
```

📊 [submit_voting_xgbtuned.csv] 파일로 결과 저장

---

## 🛠 사용된 패키지

```text
pandas
scikit-learn
xgboost
catboost
lightgbm
matplotlib
numpy
```

---

## 📎 프로젝트 구조

```
📁 thyroid-cancer-classification
├── train.csv
├── test.csv
├── sample_submission.csv
├── 01_modeling.py
├── submit_voting_xgbtuned.csv
└── README.md
```

---

## 📝 프로젝트 회고

- 클래스 불균형 대응 기법(scale_pos_weight, auto_class_weights)의 효과 체감
- Threshold 직접 튜닝으로 Precision-Recall tradeoff 조절 경험
- 향후 SMOTE, SHAP feature importance 시각화도 적용해볼 예정

---

## 🔗 참고 링크

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

