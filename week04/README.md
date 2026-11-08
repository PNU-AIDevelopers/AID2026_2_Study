# Week 04

## 🎯 이번 주 목표

- 『혼자 공부하는 머신러닝+딥러닝』 04-1 로지스틱 회귀와 04-2 확률적 경사 하강법의 개념 정리
- 크로스엔트로피에 로그와 음수 부호를 사용하는 이유 이해
- 시그모이드·소프트맥스로 점수를 확률로 바꾸고 KNN과 비교
- 생선 7종 분류 실습 결과를 정리하고 에포크·조기 종료 이해

## 📝 스터디 진행상황

생선 159마리의 5개 특성으로 KNN과 로지스틱 회귀를 비교함. 04-1은 원문에 실습 결과가 있으며, 04-2는 개념과 예시 코드가 정리되어 있음.

- 일시·참석 인원·팀원별 활동: 원문 미기재, 추후 확인

### 진행 내용

- 손실 함수: 이진 크로스엔트로피와 로그 손실의 의미 정리
- 04-1: 표준화, KNN 기준선, 도미·빙어 이진 분류, 생선 7종 다중 분류
- 04-2: 확률적 경사 하강법, 에포크, 검증 세트를 활용한 조기 종료 개념 정리
- 확률의 표현 방식, 데이터 누수, 점수와 확률의 차이에 대한 질의응답 정리

## ✅ 진행상황

- 완료한 내용: 04-1 실습 결과 및 04-2 개념·예시 코드의 학습 기록 정리
- 미완료 내용: SGD 직접 실습 결과, 제출용 PDF, 활동 일시·참석자·팀원별 활동 확인
- 아래 수치는 원문에 기록된 결과이며, 이 문서 작성 과정에서 재실행한 결과는 아님.

---

## 📚 공부 내용 정리

### 1. 손실 함수와 이진 크로스엔트로피

정답을 `y ∈ {0, 1}`, 양성 클래스의 예측 확률을 `p`라고 하면 다음과 같음.

```text
L = −[y · ln(p) + (1 − y) · ln(1 − p)]
y = 1: L = −ln(p)
y = 0: L = −ln(1 − p)
```

- 정답 클래스에 낮은 확률을 부여할수록 손실이 커짐. 정답 확률이 0에 가까워지면 손실은 무한히 커짐.
- 로그는 여러 샘플의 확률 곱을 합으로 바꾸므로 최대우도 추정과 연결됨.
- 확률의 로그는 0 이하이므로 음수 부호를 붙여 최소화할 손실로 사용함.
- 정답이 1일 때 `p=0.99, 0.5, 0.1, 0.01`의 손실은 각각 약 `0.010, 0.693, 2.303, 4.605`임.
- 정확도는 가중치의 작은 변화에 민감하게 변하지 않아 경사 기반 최적화의 직접적인 목적 함수로 쓰기 어려움.
- 선형 로지스틱 회귀의 크로스엔트로피 목적 함수는 볼록하지만, 모든 신경망의 크로스엔트로피 최적화까지 볼록한 것은 아님. 수렴에는 학습률 등 조건도 영향을 줌.

### 2. 로지스틱 회귀와 확률

이름에 회귀가 들어가지만 **분류 모델**이며, 선형 점수 `z = w·x + b`를 확률로 변환함.

| 구분 | 이진 분류 | 다중 분류 실습 |
|---|---|---|
| 변환 | 시그모이드 `1 / (1 + exp(−z))` | 소프트맥스 `exp(zᵢ) / Σ exp(zⱼ)` |
| 출력 | 양성 클래스 확률 | 클래스별 확률, 합은 1 |
| 확인 | `expit(z)` | `softmax(scores, axis=1)` |

- 시그모이드는 `z=0`에서 0.5이며, `z=3.5`에서는 약 0.971임.
- `decision_function()`은 변환 전 점수, `predict_proba()`는 클래스별 확률을 반환함.
- 이진 로지스틱 회귀에서 시그모이드 결과는 `classes_[1]`의 확률과 대응함.
- 이번 다중 로지스틱 회귀에서는 소프트맥스 결과가 예측 확률과 대응함. 이 관계를 모든 분류기의 공통 규칙으로 적용해서는 안 됨.
- `C`는 규제 강도의 역수 역할을 하므로 작을수록 규제가 강함. 릿지·라쏘의 `alpha`와 방향이 반대임.

| 비교 | KNN | 로지스틱 회귀 |
|---|---|---|
| 학습 | 훈련 데이터를 저장하고 이웃 검색 | 가중치와 절편 학습 |
| 경계 | 국소적으로 비선형 가능 | 입력 특성 공간에서 선형 |
| 확률 | 이웃의 투표 비율 | 선형 점수를 변환한 연속값 |
| 유의점 | 균일 가중치에서 k=3이면 0, 1/3, 2/3, 1 | 연속 확률이라고 보정 정확도가 보장되지는 않음 |

### 3. 생선 7종 분류 실습 (04-1)

- 입력: `Weight`, `Length`, `Diagonal`, `Height`, `Width`의 5개 특성
- 타깃: `Species`, 총 159마리·7개 클래스
- 클래스별 개수: Perch 56, Bream 35, Roach 20, Pike 17, Smelt 14, Parkki 11, Whitefish 6
- `random_state=42`, 기본 테스트 비율 25%로 훈련 119마리·테스트 40마리를 분리함.
- 원문 결과를 따르기 위해 아래 분할에는 `stratify`를 추가하지 않음. 계층화 분할은 추가 실험으로 비교할 수 있음.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)
ss = StandardScaler()
train_scaled = ss.fit_transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

mask = (train_target == 'Bream') | (train_target == 'Smelt')
train_bs, target_bs = train_scaled[mask], train_target[mask]
lr_binary = LogisticRegression()
lr_binary.fit(train_bs, target_bs)
print(lr_binary.classes_)
print(lr_binary.predict_proba(train_bs[:3]))
print(expit(lr_binary.decision_function(train_bs[:3])))

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.classes_)
print(lr.predict_proba(test_scaled[:3]))
print(softmax(lr.decision_function(test_scaled[:3]), axis=1))
```

#### 원문 실습 결과

| 모델 | 설정 | 훈련 정확도 | 테스트 정확도 |
|---|---|---|---|
| KNN | k=3 | 0.8908 | 0.8500 |
| 이진 로지스틱 회귀 | Bream·Smelt | 원문 미기재 | 원문 미기재 |
| 다중 로지스틱 회귀 | C=20, max_iter=1000 | 0.9328 | 0.9250 |

이진 분류의 클래스 순서는 `['Bream', 'Smelt']`이며 첫 3개 훈련 샘플의 확률은 다음과 같음.

| 샘플 | Bream 확률 | Smelt 확률 | 점수 z |
|---|---|---|---|
| 1 | 0.998 | 0.002 | -6.03 |
| 2 | 0.027 | 0.973 | 3.57 |
| 3 | 0.995 | 0.005 | -5.27 |

다중 분류의 확률 열 순서는 `Bream, Parkki, Perch, Pike, Roach, Smelt, Whitefish`임. 첫 3개 테스트 샘플에서 가장 높은 확률은 각각 Perch 0.842, Smelt 0.946, Pike 0.934였음. 반올림한 확률 합은 1과 조금 다를 수 있음.

이번 분할에서는 로지스틱 회귀의 테스트 정확도가 7.5%p 높았음. 단일 분할 결과이므로 항상 더 우수하다고 일반화하지 않음.

### 4. 확률적 경사 하강법 (04-2, 개념 정리)

기울기 반대 방향으로 가중치를 조금씩 갱신하여 손실을 줄이는 최적화 방법임.

| 방식 | 한 번의 갱신에 사용하는 샘플 |
|---|---|
| 배치 경사 하강법 | 전체 훈련 세트 |
| 미니배치 경사 하강법 | 훈련 세트의 작은 묶음 |
| 확률적 경사 하강법 | 샘플 1개 |

- 손실 함수는 개별 샘플의 오차, 비용 함수는 여러 샘플의 합·평균을 가리키는 구분으로 설명할 수 있으나 문헌마다 용어를 혼용하기도 함.
- **에포크**는 훈련 세트 전체를 한 번 사용하는 주기임. 샘플 1,000개에서 배치 크기 1이면 1,000회, 32이면 마지막 작은 배치를 포함해 32회 갱신함.
- 에포크가 부족하면 과소적합할 수 있으며, 너무 오래 학습하면 과대적합할 수 있음. 반드시 발생하는 것은 아님.
- 조기 종료는 검증 성능이 개선되지 않을 때 중단하는 방법임. 테스트 세트는 최종 평가에 사용함.

#### 원문의 SGD 예시 코드

아래는 위 실습의 변수들이 준비된 상태에서 사용하는 예시이며, 원문에는 실행 점수가 기재되어 있지 않음.

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(
    loss='log_loss', max_iter=1000, tol=1e-3,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=5, random_state=42
)
sgd.fit(train_scaled, train_target)
print(sgd.score(train_scaled, train_target))
print(sgd.score(test_scaled, test_target))
```

- `loss='log_loss'`는 로지스틱 손실, `loss='hinge'`는 선형 SVM의 힌지 손실을 사용함.
- **원문 설명 보완:** `SGDClassifier(early_stopping=True)`는 검증 손실 대신 검증 정확도인 `score()`의 개선 여부를 사용함. 최상 가중치 자동 복원을 전제로 해석하지 않음. [공식 API 설명](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- 원문 코드는 내부 검증 분리 전에 표준화를 수행함. 검증 세트까지 엄격히 분리하려면 먼저 훈련·검증 데이터를 나눈 뒤 훈련 부분만으로 스케일러를 학습시키는 별도 절차가 필요함.

### 5. 질문 및 정리

**Q1. KNN의 확률은 왜 거친 값으로 나올까?**

균일 가중치에서 k개 이웃의 투표 비율로 계산하기 때문임. k=3이면 네 가지 값만 가능함. 거리 가중치를 적용하면 같은 제약이 사라지지만 확률의 신뢰성이 자동 보장되는 것은 아님.

**Q2. 점수와 확률은 어떻게 다를까?**

점수 z는 실수 범위의 선형 출력이고, 확률은 시그모이드 또는 소프트맥스로 변환한 결과임. 확률 열은 반드시 `classes_`와 함께 해석함.

**Q3. 테스트 세트에서 스케일러를 다시 fit하면 왜 안 될까?**

평가 데이터의 통계량을 사용하고 훈련 때와 다른 기준으로 변환하기 때문임. 평가가 왜곡되며 반드시 점수가 높아지는 것은 아님. 훈련 세트에서 학습한 스케일러의 `transform()`을 재사용함.

**Q4. 조기 종료에서 검증 성능을 보는 이유는 무엇일까?**

훈련 데이터에 대한 성능 향상이 새로운 데이터의 성능 향상을 보장하지 않기 때문임. 검증 성능으로 중단 시점을 결정하며, 감시 지표와 최상 가중치 복원 여부는 구현별로 확인해야 함.

### 공유 자료 및 실습 노트북

- 교재: 혼자 공부하는 머신러닝+딥러닝
- [4주차 스터디 원문 및 그래프](https://rlaguswl25808134-source.github.io/AID-MLB-4week/)
- [04-1 생선 분류 실습 노트북](https://github.com/rlaguswl25808134-source/AID-MLB/blob/MLB-4-1/4-1.ipynb)

---

## 💬 스터디 회고

원문 회고에서는 KNN의 이웃 투표와 로지스틱 회귀의 확률 변환을 비교하면서 모델별 확률 표현의 차이를 이해했다고 정리함. 특히 `decision_function()`과 `predict_proba()`를 직접 비교해 점수와 확률의 연결을 확인함.

## ⏭️ 다음 주 계획

- SGDClassifier를 직접 실행하여 에포크에 따른 성능 변화를 확인
- 교차 검증과 정밀도·재현율을 추가로 학습
- 구체적인 다음 주 활동 범위는 팀 논의 후 확정
