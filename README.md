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

---

## 🔧 전처리 및 특성 선택

- Label Encoding을 통해 범주형 변수 처리
- `Gender`, `Smoke`, `Weight_Risk`, `Diabetes`는 feature 중요도가 낮아 제거
- 학습/검증 세트를 80:20 비율로 분리 (`stratify` 적용)

---

## 🤖 모델링 및 앙상블

### 🎯 XGBoost (튜닝 포함)

```python
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid = {
    'max_depth': [6],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300],
    'scale_pos_weight': [1, 7.3, 15]
}
rs_xgb = RandomizedSearchCV(xgb, param_grid, scoring='f1', n_iter=10, cv=5)
rs_xgb.fit(X_train, y_train)
best_xgb = rs_xgb.best_estimator_
```

📌 **설명**:  
XGBoost 모델에 대해 학습률, 트리 깊이, 클래스 가중치 등을 랜덤 탐색을 통해 최적화합니다.  
클래스 불균형 문제는 `scale_pos_weight`로 조정합니다.

---

### 🧩 Soft Voting 앙상블

```python
from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(
    estimators=[('xgb', best_xgb), ('cat', best_cat), ('lgb', lgb_model)],
    voting='soft'
)
voting_model.fit(X_train, y_train)
```

📌 **설명**:  
XGBoost, CatBoost, LightGBM 세 가지 모델의 예측 결과를 평균내어 보다 안정적인 예측 성능을 유도합니다.  
Soft Voting은 `predict_proba()`를 기반으로 확률 평균을 계산합니다.

---

### 🔍 Threshold 튜닝

```python
from sklearn.metrics import f1_score
import numpy as np

proba = voting_model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.30, 0.71, 0.01)
f1_scores = [f1_score(y_val, proba > t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
```

📌 **설명**:  
기본 threshold인 0.5 대신, F1-score가 가장 높은 threshold를 직접 탐색합니다.  
이는 클래스 불균형 상황에서 Precision/Recall 균형을 잡는 데 매우 중요합니다.

---

## 📈 결과 및 평가

- **최종 검증 정확도(Accuracy)**: 88.0%
- **클래스 1 (암)의 F1-score**: 0.4688
- Confusion Matrix 및 Classification Report 분석 포함

### 🎨 F1-score vs Threshold 시각화

![F1 vs Threshold](images/result.png)

> 위 그래프는 다양한 threshold에서의 F1-score 변화를 시각화한 것으로, 최적 기준점을 선택하는 데 활용됩니다.

---

## 🛠 사용된 패키지

```text
pandas, scikit-learn, xgboost, catboost, lightgbm, matplotlib, numpy
```

---

## 📝 프로젝트 회고

- 클래스 불균형 대응 전략(scale_pos_weight, auto_class_weights)의 효과 체감
- 단일 모델보다 앙상블 + threshold 최적화의 성능 우위 확인
- 향후 SHAP 해석 및 SMOTE 기반 증강도 실험할 예정

---

## 🔗 참고 링크

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

