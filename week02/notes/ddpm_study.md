
# DDPM 기초개념

$x_0$ : 원본 이미지
$x_T$ : 노이즈

$t$ : 몇 번째 노이즈 단계인가

ddpm에서 $x_T$ 는 확률 변수이다.
$x_0$ 가 고양이 사진이라고 하면, 무작위 노이즈를 추가한 $x_1$ 는 여러 가능한 값 중 하나가 랜덤으로 나오는 변수이다.


$q(x_t | x_{t-1})$ : $x_{t-1}$가 주어졌을 때 $x_t$가 나올 확률분포
예를 들어 $q(x_1 | x_0)$는 원본 이미지 $x_0$가 있을 때, 거기에 noise를 넣어 $x_1$가 나올 확률 분포이다.


Markov Chain : 미래의 상태가 오직 현재의 상태에 의해서만 결정되고, 과거의 역사와는 독립적이라는 마르코프 성질을 만족하는 이산시간 확률 과정

이를 DDPM에 적용하면,
$q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})$가 된다.
즉, $x_t$를 만들 때 과거의 모든 이미지를 볼 필요 없이 직전 이미지 $x_{t-1}$만 있으면 된다.

따라서 논문의 식은 다음과 같이 된다.
$q(x_{1:T} | x_0) = \prod_{t=1}^{T}q(x_t | x_{t-1})$
이는 $x_0$에서 시작했을 때 $x_1, x_2, ... x_T$ 전체 과정의 확률을 의미한다.


$\prod_{t=1}^{T}q(x_t | x_{t-1})$는 어디서 나왔는가?
이는 조건부 확률의 chain rule 때문이다.

[조건부 확률 구체적인 증명](https://blog.naver.com/mykepzzang/220834907530)
$P(A, B, C) = P(A) P(B|A) P(C|A,B) = P(A) P(B|A) P(C|A,B)$ 이다.
따라서 $q(x_1, x_2, ... x_T | x_0) = q(x_1|x_0) q(x_2 | x_0, x_1), ...$ 이고, 여기서 Markov property를 적용해보자.

[마르코프 체인 구체적인 증명](https://en.wikipedia.org/wiki/Markov_property)
$P(Xn+1​∣Xn​)=P(Xn+1​∣Xn​,Xn−1​,…,X0​)$이므로 
$q(x_1, x_2, ... x_T | x_0) = q(x_1 | x_0)q(x_2 | x_1) q(x_3 | x_2)...q(x_T|x_{T-1})$이다.
따라서 최종적으로 초기 등장했던
$q(x_{1:T} | x_0) = \prod_{t=1}^{T}q(x_t | x_{t-1})$식이 성립하게 된다.


$\mathcal N(\mu,\sigma^2)$는 정규 분포인 Gaussian distribution이다.
예로,
$X\sim \mathcal N(0,1)$는 평균 0, 분산 1인 정규분포에서 값을 뽑는다는 뜻이다.

DDPM에서는 각 단계에 Gaussian noise를 넣는다.
$q(x_t|x_{t-1})=\mathcal N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)$

즉, $x_t$는 평균이 $\sqrt{1-\beta_t}x_{t-1}$이고, 분산이 $\beta_t I$인 가우시안 분포에서 샘플링된다!

#### 노이즈에 대한 구체적인 설명

이미지는 vector이다.
x_{t-1}을 이전 t-1 단계 이미지라 하고, 각 픽셀에 $\sqrt{1-\beta_t}$를 곱한다. 즉, 원래 이미지의 영향력을 조금씩 줄인다!
$\beta_t I$는 모든 픽셀 방향에 같은 크기의 독립적인 Gaussian noise를 넣기 위해 정의한 것이다.
$\beta_t$의 값은 noise의 세기이다. 이 값이 작으면 원본에 가깝고, 크면 $x_{t-1}$의 값이 많이 줄어든다.

예를 들어 $\beta_t I = 0.1$이면,
$x_t = 0.949 x_{t-1} + 0.316ϵ$
즉, 원본의 값을 0.949만큼 줄이고 0.316만큼 랜덤한 노이즈를 추가한다!

여기서 등장하는 입실론, ϵ는 Gaussian noise이다.
$ϵ \sim \mathcal N(0, 1)$이고, $\mathcal N(\mu,\sigma^2)$는 $X = \mu \sigma + \epsilon$이다.

그럼, $x_t$를 만들기 위해 앞선 noise 이미지를 모두 계산해야 하는가?
마르코프 체인만 보면 그렇다.
그러나 DDPM에서는 Gaussian의 특별한 성질 덕분에, $x_0 -> x_t$를 한 번에 계산할 수 있다!


#### 한번에 구해보자!

확률 분포 샘플링 식은 다음과 같다.
$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t$

여기서 $\alpha_t = 1 - \beta_t$라고 하자.
그럼 식은 $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t$이다.
즉, $\beta_t$가 noise의 세기였다면, $\alpha_t$는 기존 이미지를 얼마나 남길지를 의미한다.

이제 $x_1$식을 $x_2$식 안에 넣어보자.
$x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{1 - \alpha_1}\epsilon_1) + \sqrt{1 - \alpha_2}\epsilon_2$
$x_2 = \sqrt{\alpha_2 \alpha_1}x_0 + \sqrt{\alpha_2(1 - \alpha_1)}\epsilon_1 + \sqrt{1 - \alpha_2}\epsilon_2$
...
다음과 같이 전개하면 $\alpha_2\alpha_1$와 같은 항이 계속해서 이어지게 된다.

$\alpha_1\alpha_2 ... \alpha_t = \bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$라고 정의하자.
예로, $\bar{\alpha}_3 = \alpha_1\alpha_2\alpha_3$이다.
직관적으로 't단계까지 왔을 때 원본 신호가 누적해서 얼마나 남았는가'를 의미한다.

이때, $x_0$가 없는 나머지 noise항은 Gaussian noise이므로
독립인 Gaussian을 선형 결합하면 결과 또한 Gaussian이 된다.

$\epsilon_1 \sim \mathcal N(0, 1)$, $\epsilon_2 \sim \mathcal N(0, 1)$라고 할 때,
$A\epsilon_1 + B\epsilon_2$은 
$E[Y] = E[A\epsilon_1 + B\epsilon_2] = AE[\epsilon_1] + BE[\epsilon_2] = 0$
$Var(Y) = Var(A\epsilon_1 + B\epsilon_2) = A^{2}Var(\epsilon_1) + B^{2}Var(\epsilon_2) + 2ABCov(\epsilon_1 \epsilon_2) = A^{2} + B^{2}$
둘은 독립이므로 공분산은 0이다.

즉, $Y = A\epsilon_1 + B\epsilon_2$는 $Y \sim \mathcal N(0, A^{2} + B^{2})$이다.
그리고 위 식은 다시 $\epsilon \sim \mathcal N(0, 1)$에 대하여
$Y = \sqrt{A^{2} + B^{2}}$이다.

두 값이 항상 같다는 것이 아니라, 확률분포가 동일하다!

이를 다시 DDPM 식에 적용하면,
$\sqrt{\alpha_2(1 - \alpha_1)}\epsilon_1 + \sqrt{1 - \alpha_2}\epsilon_2$를 $\sqrt{1 - \alpha_1\alpha_2}\epsilon$로 표현할 수 있다.
그리고 $\bar{\alpha}_2 = \alpha_1\alpha_2$이므로,
$x_2 = \sqrt{\alpha_2 \alpha_1}x_0 + \sqrt{1 - \alpha_1\alpha_2}\epsilon$가 된다.

이를 일반화하면,
$x_t = \bar{\alpha}_t x_0 + \sqrt{1-\bar{a}_t} \epsilon$이고, $\epsilon \sim \mathcal N(0, 1)$ 이다.

오해하지 말아야 할 것은 개별 샘플은 다르지만 분포가 같다는 것.


결론적으로 우리는 $q(x_t | x_0) = \mathcal N(x_t; \bar{\alpha}_t, 1-\bar{a}_t)$를 쓸 수 있다.
원본 $x_0$을 알고 있다면 원하는 timestep t의 noisy image $x_t$를 바로 샘플링 할 수 있다!


# DDPM 학습의 핵심

그럼 noisy image $x_t$를 다시 원본 방향으로 되돌리려면 대체 무엇을 학습해야 하는가?
우리가 원하는 것은 사실 $x_t \to x_{t-1}$이다. (reverse process)
DDPM에서는 이를 $p_\theta(x_{t-1} | x_t)$로 학습한다.
이는 현재 noisy image $x_t$가 주어졌을 때, 그보다 한 단계 덜 noisy한 $x_{t-1}$의 분포를 의미한다.

$p_\theta(x_{t-1} | x_t) = \mathcal N(x_{t-1}, \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))$
위 식에서 필요한 것은 $\mu$ 평균과 $\sigma$ 분산이다.

논문의 주 실험에서는 $\Sigma_\theta (x_t, t)$를 timestep별 고정값으로 둔다.
그래서 실질적으로 남는 문제는 $\mu_\theta (x_t, t)$, Reverse Gaussian의 평균을 어떻게 정할 것인가?(어느 방향으로 가야 원래 이미지에 가까워지는가?) 이다.

신기하게도 DDPM은 최종적으로 모델이 noise $\epsilon$을 예측하도록 한다.

우리가 $x_t$를 만들 때 forward 식에서
$x_t = \bar{\alpha}_t x_0 + \sqrt{1-\bar{a}_t} \epsilon$ = 원본 성분 + noise 성분으로 나눈다.
즉, 모델은 $x_t$를 보고 이 안에서 noise가 어떤 부분인지 알아내면 된다.

우리는 학습할 때 정답 $\epsilon$을 알고 있다.
그러므로 모델에게 $x_t$에 대하여 현재 timestep t의 noise를 예측하게 한다.
모델이 예측한 값을 $\mu_\theta (x_t, t)$라 하자.
실제 $\epsilon$과 그 값의 차이를 loss로 하면,
$Loss = |\epsilon-\epsilon_\theta(x_t,t)\|^{2}$이다.

즉, 모델인 U-net은 입력으로 $x_t, t$를 받고 출력으로 $\epsilon_\theta(x_t,t)$를 내놓는다.
논문에서는 timestep 정보를 sinusoidal position embedding으로 네트워크에 넣는다.


그래서 noise를 예측한 후 어떻게 $x_{t-1}$로 갈 수 있을까?

모델로부터 noise값인 $\epsilon_\theta(x_t,t)$를 얻었다고 하자.
이를 이용해 Reverse Process의 평균을 계산할 수 있다.
$\mu_\theta (x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t} {\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t,t) \right)$

즉, noise를 잘 예측하면 이를 이용해 $x_{t-1}가 있어야 할 위치를 계산할 수 있다.


# 실제 DDPM 학습 알고리즘

Step 1
데이터셋에서 원본 이미지 $x_0$을 가져온다.

Step 2
$t\sim Uniform(1,\dots,T)$에서 t를 하나 뽑는다.

Step 3
$\epsilon \sim \mathcal N(0, 1)$를 하나 뽑는다.

Step 4
$x_t = \bar{\alpha}_t x_0 + \sqrt{1-\bar{a}_t} \epsilon$를 만든다.

Step 5
모델에 넣어 $\epsilon_\theta(x_t,t)$를 구한다.

Step 6
$Loss = |\epsilon-\epsilon_\theta(x_t,t)\|^{2}$를 구한다.

Step 7
Gradient Descent를 한다.

중요한 점은 매번 랜덤하게 t를 뽑아서 $x_0 \to x_t$를 만든 다음
$x_t$에 들어간 noise를 맞추도록 학습한다.

최종 식 미리보기
$L_{\text{simple}} = E_{t,x_0,\epsilon} \left[ \left\| \epsilon - \epsilon_\theta \left( \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t \right) \right\|^2 \right]$
정답 noise와 모델이 예측한 noise의 MSE

Sampling
U-net이 $x_T ~ \mathcal N(0, 1)$에서 시작해서
$x_T$에서부터 $x_0$까지 학습된 noise를 덜어내며 이미지를 생성함.

# Loss Function 이해하기

최종 Loss 식에 대해 알아보자.
우선 학습할 평균 $\mu_\theta (x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t} {\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t,t) \right)$가 왜 이렇게 되는지
즉, $\frac{1}{\sqrt{\alpha_t}}$, $\beta_t$, $\sqrt{1-\bar\alpha_t}$가 왜 튀어나오는지 그 이유를 알아보자.
출발점 : $x_t$에서 $x{t-1}$로 돌아갈 때 $x_{t-1}$이 어느 위치에 있을 가능성이 가장 높은가?
위치 = 평균

# 결론 : 어렵다

Diffusion의 기초가 되는 VAE, GAN, diffusion model 우선 정리하고 DDPM을 다시 복습해보겠다..
