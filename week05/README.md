# Week 05

## 🎯 이번 주 목표

- 『혼자 공부하는 머신러닝+딥러닝』 05-1 결정 트리와 05-2 교차 검증과 그리드 서치 정리
- 지니 불순도·엔트로피와 정보 이득으로 분할 기준을 선택하는 원리 이해
- 트리의 깊이를 제한하여 과대적합을 줄이고 특성 중요도 해석
- 검증 세트와 교차 검증을 활용해 그리드 서치·랜덤 서치 수행 과정 이해

## 📝 스터디 진행상황

알코올 도수·당도·pH를 이용해 와인 6,497개를 레드(0)와 화이트(1)로 분류하는 원문 실습과 개념을 정리함.

- 일시·참석 인원: 11/12(목) 참석 인원 : 성준서, 정예빈, 김영빈

### 진행 내용

- 05-1: 결정 트리의 규칙, 노드 구조, 불순도와 정보 이득, 깊이 제한, 특성 중요도
- 05-2: 훈련·검증·테스트 세트의 역할, 계층화 교차 검증, 하이퍼파라미터 탐색
- 와인 분류: 로지스틱 회귀와 결정 트리의 성능 비교 및 랜덤 서치 결과 정리

## ✅ 진행상황

- 완료한 내용: 05-1·05-2 개념, 예시 코드, 원문 실습 결과 정리
- 아래 성능 수치는 원문에 기록된 결과이며, 이 문서 작성 과정에서 재실행한 결과는 아님.

---

## 📚 공부 내용 정리

### 1. 로지스틱 회귀와 결정 트리

로지스틱 회귀는 특성의 가중합을 확률로 변환함. 결정 트리는 특성값에 대한 조건을 반복 적용하여 데이터를 나누므로 규칙을 따라 예측 과정을 설명할 수 있음.

| 구분 | 의미 |
|---|---|
| 루트 노드 | 모든 훈련 샘플에서 출발하는 첫 노드 |
| 분할 노드 | 특성과 임곗값을 이용해 데이터를 나누는 노드 |
| 가지 | 조건의 참·거짓에 따라 이동하는 경로 |
| 리프 노드 | 분할을 끝내고 예측을 결정하는 노드 |

와인 원문에서는 로지스틱 회귀의 테스트 정확도가 0.778, 제한 없는 결정 트리는 0.8592였음. 이 결과를 모든 데이터에서 결정 트리가 우수하다는 의미로 일반화하지 않음.

### 2. 불순도와 정보 이득

불순도는 한 노드에 서로 다른 클래스가 얼마나 섞여 있는지 나타냄. 클래스 비율을 pᵢ라고 하면 다음과 같음.

```text
지니 불순도 = 1 − Σ pᵢ²
엔트로피 = −Σ pᵢ log₂(pᵢ)  (pᵢ=0인 항은 0으로 처리)
정보 이득 = 부모 불순도
          − (왼쪽 샘플 수 / 부모 샘플 수) × 왼쪽 불순도
          − (오른쪽 샘플 수 / 부모 샘플 수) × 오른쪽 불순도
```

- 이진 분류에서 지니 불순도는 두 클래스가 반반이면 0.5, 한 클래스만 있으면 0임.
- 원문 루트의 레드 1,258개·화이트 3,939개로 계산하면 지니는 약 0.367, 엔트로피는 약 0.798임.
- 왼쪽 2,922개·불순도 0.481, 오른쪽 2,275개·불순도 0.069로 나누면 정보 이득은 약 0.066임.
- 각 노드에서 불순도를 가장 많이 줄이는 분할을 선택함. 이 국소적인 선택이 전체 트리의 전역 최적성을 보장하는 것은 아님.
- 원문은 기본 지니 기준을 사용하며, `criterion='entropy'`로 엔트로피 기준을 선택할 수 있음.

### 3. 깊이 제한과 특성 중요도

- 제한 없는 트리는 훈련 정확도 0.9969에 비해 테스트 정확도 0.8592로 차이가 큼.
- `max_depth=3`으로 성장 깊이를 제한하면 훈련 0.8455, 테스트 0.8415로 격차가 줄어듦. 테스트 점수 자체는 약 1.77%p 낮아졌으므로 성능 향상과 구분해야 함.
- 최대 깊이를 미리 제한하는 방법은 사전 가지치기에 해당함.
- `plot_tree(dt, max_depth=1)`은 그림에 표시할 깊이만 제한함. 학습 모델의 깊이를 제한하는 `DecisionTreeClassifier(max_depth=3)`과 다름.
- 결정 트리는 특성값의 순서에 따라 분할하므로 일반적으로 표준화가 필요하지 않음. 원본 단위를 사용하면 조건을 읽기 쉬움.

깊이 3인 원문 트리의 `feature_importances_`는 다음과 같음. 해당 모델의 불순도 감소 기여도를 나타내며 인과관계를 뜻하지 않음.

| 특성 | 중요도 |
|---|---|
| alcohol | 0.12345626 |
| sugar | 0.86862934 |
| pH | 0.00791440 |

### 4. 와인 분류 실습 (05-1)

전체 6,497개를 훈련 5,197개와 테스트 1,300개로 분리함. 아래 코드는 원문의 분할과 설정을 유지하면서 모델별 변수명을 구분한 정리 코드임.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

dt_full = DecisionTreeClassifier(random_state=42)
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
for model in (dt_full, dt_pruned):
    model.fit(train_input, train_target)
    print(model.score(train_input, train_target))
    print(model.score(test_input, test_target))

print(dt_pruned.feature_importances_)
plt.figure(figsize=(16, 10))
plot_tree(dt_pruned, filled=True,
          feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

### 5. 검증 세트와 교차 검증 (05-2)

테스트 점수를 보며 설정을 반복 선택하면 테스트 데이터에 맞춘 모델을 고르게 됨. 하이퍼파라미터 선택에는 검증 데이터를 사용하고 테스트 데이터는 최종 평가용으로 남겨 둠.

```python
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)
print(sub_input.shape, val_input.shape)  # (4157, 3), (1040, 3)
```

**원문 설명 보완:** 전체의 80%를 훈련용으로 나눈 뒤 그중 20%를 검증용으로 떼므로, 전체 기준 비율은 약 **64%·16%·20%**임. 원문 도식의 60%·20%·20%와 코드가 일치하지 않음.

교차 검증은 훈련 데이터를 여러 폴드로 나눠 각 폴드가 검증 역할을 한 번씩 맡도록 학습하고 점수를 평균함.

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

# 앞의 깊이 제한 모델과 혼동하지 않도록 평가할 모델을 명시
dt_cv = DecisionTreeClassifier(random_state=42)
scores = cross_validate(dt_cv, train_input, train_target, cv=5)
print(np.mean(scores['test_score']))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt_cv, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```

- 분류기의 기본 교차 검증은 클래스 비율을 고려한 계층화 분할을 사용함.
- `test_score`는 여기서 검증 폴드 점수이며, 따로 남긴 최종 테스트 세트 점수가 아님.
- 원문은 5-폴드 평균 0.8553, 섞은 10-폴드 평균 0.8574를 제시함. 다만 앞부분에서 `dt`를 깊이 3 모델로 재사용하므로 코드의 모델 설정을 확인할 필요가 있음. 위 코드는 깊이 제한 없는 모델을 명시한 정리 예시이며 수치 재현을 검증한 것은 아님.

### 6. 그리드 서치와 랜덤 서치

모델 파라미터는 학습으로 정해지는 분할 특성·임곗값 등이고, 하이퍼파라미터는 사용자가 정하는 깊이·최소 샘플 수 등임. 하이퍼파라미터는 서로 영향을 주므로 조합으로 탐색함.

| 설정 | 역할 |
|---|---|
| `max_depth` | 트리의 최대 깊이 |
| `min_samples_split` | 노드를 분할하기 위한 최소 샘플 수 |
| `min_samples_leaf` | 리프에 남아야 하는 최소 샘플 수 |
| `min_impurity_decrease` | 분할에 필요한 최소 가중 불순도 감소량 |

#### 그리드 서치

```python
from sklearn.model_selection import GridSearchCV

grid_params = {
    'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
    'max_depth': range(5, 20),
    'min_samples_split': range(2, 100, 10),
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    grid_params, cv=5, n_jobs=-1)
grid.fit(train_input, train_target)
print(grid.best_params_)
print(grid.best_score_)
```

- 9 × 15 × 10 = 1,350개 조합을 평가하며 5-폴드에서 6,750회 학습함. 기본 재학습까지 포함하면 최적 모델 학습이 1회 추가됨.
- `best_params_`는 최적 조합, `best_score_`는 그 조합의 평균 검증 점수임.
- 기본 `refit=True`에서 `best_estimator_`는 전달한 전체 훈련 세트로 다시 학습한 최적 모델임.
- `cv_results_['mean_test_score']`로 각 조합의 평균 검증 점수를 확인함.

#### 랜덤 서치

```python
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

random_params = {
    'min_impurity_decrease': uniform(0.0001, 0.001),
    'max_depth': randint(20, 50),
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 25),
}
search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42), random_params,
    n_iter=100, cv=5, n_jobs=-1, random_state=42
)
search.fit(train_input, train_target)
print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_.score(test_input, test_target))
```

- 후보를 분포에서 뽑아 100개 조합을 평가하므로 5-폴드에서 500회 학습하고 최적 모델을 다시 학습함.
- `uniform(0.0001, 0.001)`의 두 번째 인수는 상한이 아닌 구간 폭임. 범위는 0.0001부터 0.0011까지임.
- `randint(20, 50)`은 20 이상 50 미만의 정수를 뽑음.
- 원문의 최적 조합: `max_depth=39`, `min_impurity_decrease≈0.0003410`, `min_samples_leaf=7`, `min_samples_split=13`.
- 원문 최고 검증 정확도는 0.8695, 최종 테스트 정확도는 0.8600임. 테스트 점수가 검증 점수보다 항상 낮아야 하는 것은 아님.

### 7. 원문 실습 결과 비교

| 모델 | 설정 | 훈련 정확도 | 테스트 정확도 |
|---|---|---|---|
| 로지스틱 회귀 | 표준화 + 기본 설정 | 0.781 | 0.778 |
| 결정 트리 | 깊이 제한 없음 | 0.9969 | 0.8592 |
| 결정 트리 | max_depth=3 | 0.8455 | 0.8415 |
| 랜덤 서치 최적 트리 | n_iter=100 | 원문 미기재 | 0.8600 |

로지스틱 회귀 수치는 원문의 비교 표에 제시된 값임. 모델 선택에는 검증 점수를 사용하며, 위 테스트 점수 표는 원문 결과를 비교하기 위한 기록임.

### 공유 자료

- 교재: 혼자 공부하는 머신러닝+딥러닝
- [5주차 스터디 원문·설명 그림·현장 사진](https://rlaguswl25808134-source.github.io/AID-MLB-5week/)
- [와인 실습 데이터](https://bit.ly/wine_csv_data)

---

## 💬 스터디 회고

이번 학습의 핵심은 결정 트리의 분할 규칙을 해석하고, 훈련 점수만으로 모델을 평가하지 않는 것임. 깊이를 제한하면 모델이 단순해지지만 테스트 정확도가 반드시 높아지지는 않으며, 적절한 설정은 교차 검증으로 선택해야 함. 개인별 회고는 원문 미기재로 추후 보완함.

## ⏭️ 다음 주 계획

- 저장소의 6주차 계획에 따라 교차 검증·탐색을 복습하고 랜덤 포레스트·엑스트라 트리·GBM 학습 준비