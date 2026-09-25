# 학습효과를 검사하며 teacher 패치를 제거하는 방법

작성일: 2026-09-26. 상태: 문헌 기반 연구 설계안. 새 모델의 학습·성능 검증은 아직 하지 않았다.

후속: 이 설계의 정확한 검사 버전을 별도 실행기에 구현했다. 현재 PC의 자원 제한·bounded candidate search·실행 명령은 [RTX 5080 실행 안내](LOCAL_DELETION_5080.md)를 따른다. 실제 COCO 학습은 아직 실행하지 않았다.

기존 서버에서는 [A5000 서버 실행 안내](SERVER_DELETION.md)를 따른다. 기존 Base teacher와 CUDA12.4 환경을 재사용하고, 별도 결과 폴더에서 네 방법을 순차 비교한다.

## 1. 제안하는 연구 질문

**현재 학생에게 줄 학습 신호를 유지하면서 teacher 입력에서 제거할 수 있는 패치 집합을 찾을 수 있는가?**

같은 기준으로 두 방법을 설계한다.

- **탐색 후 제거:** Random10이 가져온 새 패치를 먼저 포함한 뒤, 기존 패치 중 역할이 줄어든 것을 제거한다.
- **자동 중단 제거:** 무작위 추가 없이 전체 패치에서 시작해, 학습효과 검사를 통과하는 만큼만 제거한다.

여기서 “불필요”는 객체나 세계에 대한 절대적 속성이 아니다. **정해진 teacher·현재 student·입력·남은 패치 집합·검사 기준에서 제거 영향이 허용 범위인 상태**를 뜻한다. 실제 최종 정확도나 모든 배경 변화에서의 무해함을 보장하지 않는다. 제거 후보의 단순 배경 여부와 낮은 attention만으로는 이 판정을 내리지 않는다.

## 2. 문헌과 겹치는 부분

| 1차 문헌 | 실제 핵심 | 이번 설계에서 가져오거나 피할 점 |
| --- | --- | --- |
| [MaskedKD, ECCV 2024](https://maskedkd.github.io/) | Student attention으로 frozen teacher의 입력 패치를 줄인다. 초기의 덜 정확한 teacher supervision도 curriculum 효과를 낼 수 있다. | Student는 full input을 유지한다. Teacher 정확도 최대화와 student 학습 개선을 구별한다. |
| [Sufficient Input Subsets, AISTATS 2019](https://proceedings.mlr.press/v89/carter19a.html) | 예측 confidence를 유지하는 작은 입력 부분집합을 backward selection으로 찾는다. | “지워도 답이 같으면 버린다”와 greedy deletion 자체는 신규 기여가 아니다. |
| [Overinterpretation, NeurIPS 2021](https://arxiv.org/abs/2003.08907) | 의미 있는 객체가 거의 남지 않아도 모델이 높은 confidence로 예측하는 충분 입력 부분집합을 보인다. | 출력 불변성을 의미적·인과적 안전성으로 해석하면 안 된다. |
| [DynamicViT, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf) | 학습한 중요도 모듈로 토큰을 줄이고 출력·특징 증류로 성능을 유지한다. | 학습형 selector와 teacher 출력 보존은 이미 존재한다. |
| [EViT, ICLR 2022](https://arxiv.org/abs/2202.07800) | 높은 attention 토큰을 보존하고 나머지는 하나로 융합한다. | 낮은 attention 토큰의 직접 폐기와 정보 병합은 다르다. 낮은 점수는 삭제 허가가 아니다. |
| [ToMe, ICLR 2023](https://arxiv.org/abs/2210.09461) | 유사한 토큰을 크기 정보를 반영해 병합한다. | 유사도는 중복 후보 탐색에 유용하지만, 유사한 토큰의 직접 삭제를 보증하지 않는다. |
| [DFR/Last Layer Re-Training, ICLR 2023](https://arxiv.org/pdf/2204.02937) | FG-only 진단으로 core feature가 학습됐음을 보이고 마지막 분류층 재학습으로 활용을 개선한다. | FG-only 평가 이득을 FG-only backbone 학습의 이득으로 옮겨 해석하지 않는다. |
| [Right for the Right Reasons, IJCAI 2017](https://arxiv.org/abs/1703.03717) | 사람이 irrelevant라고 표시한 입력 영역에 대한 gradient를 억제한다. | “배경을 무시하도록 학습” 자체도 기존 접근이다. |
| [Counterfactual and Invariant Data Generation, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.html) | 전경·배경 개입으로 분류기의 shortcut 의존을 줄인다. | FG 고정·BG 변경은 별도 robustness 진단으로 활용한다. |
| [LeaF, NeurIPS 2025](https://arxiv.org/html/2506.07851v2) | LLM에서 teacher/student 입력 gradient를 비교해 방해 토큰을 찾고, 원본·제거 입력에 대해 증류한다. | 가장 가까운 충돌이다. “학생 맞춤 gradient 기반 토큰 제거 KD”라는 넓은 설명만으로 novelty를 주장할 수 없다. |
| [Pareto Testing, ICLR 2023](https://openreview.net/pdf?id=cyg2YXn_BqF) | 별도 calibration과 가설검정으로 pruning 등의 정확도·비용 위험을 통제한다. | 삭제 임계값을 검증 데이터에서 정하는 절차는 근거가 있으나, 이것도 기존 기술이다. 고정 모델의 통계적 보장을 학습 중 변하는 student에 자동 적용할 수 없다. |

문헌에서 가져온 구성요소와 아래의 새 조합 제안을 구분한다. 이번 검색에서 아래 전체 절차와 동일한 방법은 확인하지 못했지만, 최초성이나 출판 가능성이 입증된 것은 아니다.

## 3. 기존 COCO 결과가 주는 제한

실험은 COCO 2017의 10종 단일 라벨 subset이다. 표준 COCO detection 성능이 아니다. Student는 항상 196개 spatial token을 보고, teacher만 패치를 제거한다. CLS는 아래 개수에서 제외한다.

DeiT-S teacher, scratch student, seed 0/1/2, epoch 100의 평균:

| 방법 | Test overall (%) | Test macro (%) |
| --- | ---: | ---: |
| MaskedKD top98 | 60.4753 | 59.9515 |
| Random10 | 61.2883 | 60.7652 |
| FG Rescue10 | 59.6623 | 59.5109 |

출처: [test_results.csv](../reports/20260923_152152_732466/analysis/test_results.csv). FG Rescue는 최대 10개, Random10은 정확히 10개를 교체하므로 전경 여부만 통제한 비교는 아니다. 그래도 “전경을 더 보존하면 증류 성능도 오른다”는 단순 가설은 현재 결과로 지지되지 않는다. Validation-best와 epoch100 결론도 섞지 않는다.

## 4. 공통 기준: 제거 후 가상의 학습 한 걸음 검사

### 4.1 단순 출력 거리의 한계

Teacher의 답 하나만 유지해도 클래스 간 soft target은 바뀔 수 있다. 반대로 teacher 확률벡터 변화가 동일해도, 학생이 이미 정답을 강하게 믿는지 다른 클래스로 틀리는지에 따라 학습 영향은 다르다.

단순히 `||q_C-q_B|| / (||p-q_B||+epsilon)`을 쓰면, 고정된 B에서 분모가 모든 후보에 같아 **삭제 후보의 순위는 teacher 출력 거리 순위 그대로**다. 이를 학생 맞춤 순위라고 부르지 않는다.

### 4.2 실제 학습 loss와 일치하는 정의

- z: 현재 student의 full-image logits. 같은 forward 결과를 모든 후보에 재사용한다.
- B: 제거 전 reference 패치 집합, C: 제거 후 후보 집합.
- q_B, q_C: 해당 입력을 본 frozen teacher의 temperature-scaled 확률.
- alpha: 저장소의 `kd_alpha`, tau: `temperature`, y_ls: label smoothing을 적용한 정답 분포.

현재 저장소 loss와 같은 reference objective를 쓴다.

```text
F_B(z) = (1-alpha) CE(y_ls, softmax(z))
         + alpha * tau^2 * KL(q_B || softmax(z/tau))

g_C = (1-alpha) (softmax(z)-y_ls)
      + alpha * tau * (softmax(z/tau)-q_C)
```

Teacher 출력과 선택 과정은 stop-gradient이다. `g_C`는 학생 logits에 대한 gradient이며, 전체 파라미터 gradient와 동일하지 않다. 현재 기본 설정은 alpha=0.5, tau=1이다.

후보별로 실제 모델을 다시 학습하지 않고, 작은 logits 벡터에서 한 번의 가상 업데이트를 계산한다.

```text
Delta_B(eta) = F_B(z) - F_B(z - eta*g_B)
Delta_C(eta) = F_B(z) - F_B(z - eta*g_C)

accept(C | B) iff
  Delta_C(eta) >= (1-epsilon) * Delta_B(eta)
  for every preregistered eta
```

예를 들어 epsilon=0.05라면 **검사한 가상 업데이트에서 reference loss 감소량의 95% 이상을 유지**한다는 뜻이다. Student 정확도 95% 보장이나 5% 이내 성능 저하 보장이 아니다. `eta={0.1,0.5,1.0}`은 시작 가능한 진단 설정이며 실제 optimizer learning rate가 아니다. Epsilon과 eta는 pilot validation으로 고정하고 test로 조정하지 않는다.

`Delta_B`가 수치 정밀도 수준으로 작으면 새로운 삭제를 허가하지 않는다. 이때 KD를 생략하면 다른 학습법이 되므로 첫 실험에는 넣지 않는다. q_C로 F_C를 만들어 비교하면 평가 target 자체가 움직이므로 반드시 동일한 F_B에서 평가한다.

첫 수치 프로토콜은 teacher를 eval 모드로 실행하고, 얻은 teacher/student logits를 float64로 올려 확률·가상 loss를 계산한다. 모든 eta의 `Delta_B >= 1e-8`을 요구하고, 부등식의 수치 허용오차는 `1e-10`으로 고정한다. 원 forward 정밀도와 이 audit 정밀도를 따로 기록한다. 이 값은 정확도 보장 계수가 아니라 수치 불안정성 처리 규칙이다.

현재 tau=1에서는 F_B가 정답·teacher 확률의 혼합 target에 대한 cross entropy와 상수항만 다르다. Gradient 식은 수치 미분과 일치했다(3-class 합성 예시 최대 절대오차 약 5.5e-11). 같은 teacher L2 변화량 0.08인 두 후보의 순위가 학생 확률에 따라 뒤집히는 합성 예시도 확인했다. **이는 수식 점검이며 COCO 성능 실험이 아니다.**

### 4.3 보장 범위와 별도 검사

검사에 통과하면 그 이미지·현재 logits·reference·검사한 eta에서 정의한 손실 감소 조건을 만족한다. 실제 AdamW 업데이트, 여러 이미지가 공유하는 파라미터, backbone 변화, 장기 학습 궤적, 새로운 배경에서의 정확도까지 보장하지 않는다.

각 패치를 하나씩 뺐을 때 통과했다는 이유로 묶음 삭제를 허가하지 않는다. 최종 C 전체를 teacher에 넣어 검사한다. 서로 대체하는 두 패치는 각각 삭제 가능해도 동시에 삭제하면 유일한 단서가 사라질 수 있다.

Teacher가 reference에서 맞았는데 후보에서 틀리는 경우를 거부하는 gate는 별도 ablation으로 둔다. 처음부터 이것을 필수로 넣으면 “학습효과 검사”와 “teacher 정확도 보존”의 효과가 섞인다. Segmentation은 주 방법의 필수 입력이 아니며, 삭제한 전경 면적과 경계·작은 객체 손실을 진단하는 데 사용한다.

## 5. 방법 A: Random10을 위한 탐색 후 제거

핵심은 **새 패치를 추가한 뒤의 teacher 신호를 reference로 삼는 순서**다.

```text
A = student attention top98
R = 원래 A 밖에서 뽑은 10개 (비교군과 짝지은 난수 제안)
B = A union R                         # 108개
D = A 안에서 고른 제거 후보 10개
C = B minus D                        # 98개
```

기존 Random10은 D도 무작위다. 제안은 `accept(C|B)`를 만족하는 D를 찾는다. 새로 들어온 R은 이 단계에서 보호한다. A만 reference로 삼으면 새 R이 제공한 정보까지 원래 출력과의 차이로 벌점화할 수 있으므로, B를 기준으로 삼는 것이 핵심이다.

현재 구현은 무작위 제거 D0를 먼저 뽑고 다음에 R을 뽑는다. 제안에서도 D0와 R을 이 순서로 미리 만들며, 탐색 중에는 기존 mask RNG를 더 소비하지 않는다. 무작위 D 후보는 D0의 앞 k개를 사용한다. 따라서 “추가를 먼저 한다”는 것은 reference 구성의 순서이며, 기존 난수 소비 순서를 바꾸라는 뜻은 아니다. 데이터 증강·dropout·mask stream은 분리한다. 같은 checkpoint의 진단에서는 A와 R을 정확히 공유한다.

버릴 후보의 순서는 낮은 student attention, 남은 패치와의 특징 유사도 등 값싼 기존 점수로 좁힐 수 있다. 다만 이 점수는 후보 생성용이다. 최종 허가는 joint audit이 내린다. 첫 연구용 구현은 정해진 후보 집합을 실제 teacher로 평가해 원리를 검증한다. 전 조합 탐색이나 전역 최적성을 주장하지 않는다.

재현 가능한 첫 pilot의 후보 생성은 다음 세 가지로 제한한다. 동일 후보를 teacher-L2 비교군에도 제공한다.

1. A에서 균등 무작위로 뽑은 D: 기존 Random10과 같은 제거 제안.
2. A에서 student 마지막 층 head-mean CLS attention이 가장 낮은 k개.
3. Student 마지막 층 patch feature의 cosine similarity로 중복 후보를 순차 제거한 k개. 현재 남은 집합에서 가장 가까운 다른 토큰과의 유사도가 큰 토큰을 먼저 제거하고, 제거할 때마다 대표 토큰이 여전히 남아 있는지 재계산한다. R은 제거 대상에서 제외한다. 동률은 낮은 attention, 낮은 원래 patch index 순으로 푼다. Student feature의 유사도가 teacher 입력 중복을 보장한다고 가정하지 않는다.

동일 mask 후보는 중복 계산하지 않는다. 통과 후보 중 `max_eta (Delta_B-Delta_C)/Delta_B`가 가장 작은 것을 선택한다. 동률은 정렬된 mask index의 사전식 순서로 푼다. 이 순위식의 분모가 작으면 앞의 작은 신호 처리 규칙을 적용한다. 후보 세 개가 모두 실패해도 모든 가능한 삭제 조합의 실패를 뜻하지 않는다.

실패 처리에는 두 가지 서로 다른 정책이 있다.

- **고정98 비교용:** 미리 정한 순서의 R에서 k개만 제안하며 k=10부터 줄여 재검사한다. B_k=A union R_k, D_k는 A의 k개, C_k는 항상98개다. 하나도 통과하지 않으면 k=0, 기존 A로 복귀한다. 각 reference가 다른 별도 제안이며 monotonicity를 가정하지 않는다. 명칭은 “최대10 교체”로 표기한다.
- **엄격한108 reference 진단용:** 108 대비 통과하는98 집합을 못 찾으면108을 유지한다. 이 경우 토큰·계산 예산이 달라지므로 고정98 성능표와 섞지 않는다.

검색 비용이 큰 첫 학습 pilot에서는 k 후보를 사전에 `{10,5,0}`으로 제한할 수 있다. 이 제한을 설정 파일에 기록하고, 전체 k 탐색 결과와 구분한다. 검사한 모든 reference와 실패 후보의 teacher 비용을 집계한다.

고정98 정책의 k=0 복귀는 108 대비 안전한 압축을 찾았다는 뜻이 아니다. **새 탐색 제안을 채택하지 않은 것**이다. Acceptance rate, k 분포, fallback 비율을 모두 보고한다. 비교군도 같은 이미지·step별 k를 사용해 “덜 교체해서 좋아진 것”을 분리한다.

k-matched 학습 비교군은 A 실험의 `(seed, epoch, sample_id, occurrence, k)` trace를 저장해 같은 데이터 순서로 재생한다. 별도 학습 중에는 student top98이 서로 달라질 수 있으므로 실제 R의 patch ID까지 같다고 주장하지 않는다. 각자 선택 밖에서 같은 난수 제안 규칙으로 k개를 뽑는다. 제거 기준 자체의 정확한 R 통제 비교는 동일 checkpoint 진단에서 수행한다. Teacher-L2와의 선택 기준 비교는 후보 생성 수와 검색 예산까지 맞춘다.

가설: Random10의 탐색 이득을 유지하면서 기존의 유용한 패치를 함께 제거하는 손해를 줄일 수 있다. 반대로 우연한 파괴가 regularization으로 유익했다면 이 방식은 더 나쁠 수 있다.

## 6. 방법 B: Random10 없는 자동 중단 제거

```text
B = 모든196개 패치                   # reference는 끝까지 고정
C = B
while len(C) > 98:
    정해진 후보 규칙으로 삭제 묶음 D를 생성
    각 C_new = C minus D를 B 대비 검사
    통과 후보 중 가상 손실이 가장 작은 C_new를 선택
    통과 후보가 없으면 더 작은 묶음으로 검사
    1개 삭제 후보도 통과하지 못하면 중단
return C
```

후보 생성·동률 처리를 deterministic하게 고정하고 random rescue를 사용하지 않는다. 직전 단계 C만 reference로 바꾸면 작은 손실이 누적되므로, 모든 단계에서 원래 B와 비교한다. 후보 검색 실패는 “제거 가능한 집합이 전혀 없다”는 수학적 증명이 아니다.

첫 pilot에서는 A의 후보 생성 규칙 중 attention과 redundancy 두 가지만 사용한다. 삭제 묶음 크기는 현재 남은 수와 목표98의 차이를 넘지 않도록 제한하며, 우선10개를 시도하고 실패하면5개,1개 순으로 줄인다. 각 크기에서 통과 후보 중 위의 worst-eta 손실 비율이 가장 작은 것을 채택한다. Student feature·attention은 같은 full-image forward에서 얻는다.

첫 프로토콜은98을 목표 하한으로 삼고, 이미지별로98–196개를 남긴다. 98에 도달하지 못하면 더 많이 남기는 것이 제거 위험 기준을 지키는 선택이다. 고정98 강제 버전은 별도 ablation이며, 위반율을 기록한다.

이 방법의 1차 목적은 **학생이 필요한 supervision을 유지하는 이미지별 teacher 계산량 조절**이다. Full teacher 신호 보존이 MaskedKD의 curriculum을 약화시킬 수 있으므로 MaskedKD보다 높은 정확도는 가설이지 전제가 아니다. 평균 token 수뿐 아니라 실제 전체 학습 시간에 맞춘 Random/attention pruning과 비교한다.

## 7. 계산량 문제와 실험 순서

정확한 검사는 teacher의 여러 forward가 필요하다. A의 한 후보만 보더라도108 reference와98 candidate를 각각 계산하면 기존98 한 번보다 비용이 크다. B의 반복 삭제도 같다. 최종 입력 token 수만으로 속도 향상을 주장하지 않는다.

1. **먼저 작은 진단:** 기존 checkpoint의 early/middle/late 고정 probe에서 삭제 제안, virtual loss, 정답→오답 전환, 전경 면적 손실을 측정한다. 동일 이미지 반복은 독립 표본으로 세지 않는다.
2. **기전 검증 pilot:** 소수 후보만 직접 검사하는 비싼 버전으로 seed0 학습을 비교한다. 시간 절약이 아니라 학습 신호 기준이 유효한지 판단한다. 효과가 없으면 selector부터 크게 만들지 않는다.
3. **효율 버전은 후속:** 효과가 있을 때 student의 이미 계산한 patch features·logits·제안 mask를 받아 삭제 집합의 검사 통과 여부/손실을 예측하는 작은 selector를 학습한다. 일부 step만 teacher로 실제 검사하며 탐색·selector 학습·검사 비용을 모두 합산한다.

세 번째는 근사 방법이다. 모든 집합을 직접 검사한 버전과 같은 per-example 검증이 있다고 표현하지 않는다. Online student 변화로 scorer가 낡을 수 있고, 고정 checkpoint에서의 calibration이 학습 전 기간에 자동 유효하지도 않다. 적어도 epoch 구간별 오차를 별도로 측정한다.

현재 teacher는 frozen이고 증강이 resize/flip이라 full-image reference는 캐시할 여지가 있다. 그러나 A의 student top98과 후보 mask는 학습 중 변하므로 teacher 출력을 무조건 재사용할 수 없다. 캐시 key에는 이미지 변환, teacher hash, 정확한 token indices를 포함해야 한다.

## 8. 최소 대조군과 판정

| 질문 | 반드시 필요한 비교 |
| --- | --- |
| Random10보다 나은가? | 기존 MaskedKD, Random10, 방법 A |
| 단순 교체량 감소 효과인가? | A와 실제 k를 맞춘 무작위 제거·추가 |
| 학생 맞춤 기준이 필요한가? | 동일 후보·동일 검색 비용에서 teacher 출력 L2만 최소화 |
| 여러 패치 상호작용 검사가 필요한가? | 패치별 영향 합만 사용 vs 최종 집합 audit |
| reference를108로 잡는 것이 중요한가? | top98 reference vs 추가 후108 reference |
| B가 효율적인가? | Full KD, 평균 token·총 계산시간을 맞춘 attention/random pruning |
| 불확실하면 멈추는 것이 중요한가? | B의 adaptive stop vs 고정98 강제 |

첫 pilot에서 모든 조합을 한꺼번에 돌리지 않는다. MaskedKD·Random10·A·B 네 조건으로 가능성을 본 뒤, 이득이 나타난 방법에 위 ablation을 적용한다. Student/teacher 초기화와 seed·데이터 순서·optimizer·LR·epoch를 맞춘다. B는 full teacher reference를 쓰므로 Full KD 결과도 해석에 포함한다.

Primary는 epoch100 test macro accuracy, secondary는 validation-best test, 원본 이미지 정확도, 목표 validation 성능까지 걸린 시간이다. 새 방법의 threshold 선택은 validation에서만 한다. 기존533 test는 여러 번 본 탐색 평가이므로 논문 수준의 확인에는 새 holdout 또는 다른 dataset을 추가한다.

동일 FG의 배경 교체 pair에서는 예측 일치도만 보면 같은 오답도 성공으로 보인다. Pair 양쪽 정답률, 원본/변경 정확도, 예측 flip, 클래스별·객체 면적별 결과를 함께 본다. COCO에서 객체 바깥을 전부 label-independent라고 가정하지 않으며 합성 경계 artifact를 통제한다. 이 평가는 shortcut 감소 주장에 필요하지만 본 가상 학습 검사가 자동으로 보장하는 결과는 아니다.

세 seed의 paired 차이와 변동을 보고하고, seed0 pilot 성공만으로 우위를 선언하지 않는다. Token 수·teacher forward 수·캐시 준비·selector 학습·감사 비용까지 포함한 GPU 시간과 wall time을 보고한다.

동일 token 예산과 동일 시간 예산은 따로 평가한다. 첫 표는 같은100 epoch·optimizer schedule에서 평균 teacher 입력 token 수를 맞춘다. 시간 비교는 고정된 학습 schedule로 얻은 validation 성능–누적 시간 곡선을 공통 시간 지점에서 비교하며, 도달한 epoch/step도 표시하고 시간에 맞춰 LR을 재조정하지 않는다. B의 가변 길이 입력은 같은 token 수끼리 teacher microbatch로 묶고 원래 minibatch의 sample 순서로 출력을 복원하는 구현을 우선한다. Padding으로 계산을 채우는 구현이라면 padded 길이를 실제 계산량에 반영한다.

## 9. 신규성 후보와 중단 기준

후보 기여는 세 가지를 함께 검증하는 데 있다.

1. 추가 후보를 포함한 reference를 먼저 만들고 기존 패치를 조건부로 제거하는 Random10의 재구성.
2. Teacher의 답 보존 대신 현재 학생의 **가상 학습 한 걸음의 효과**를 기준으로 삭제 집합을 판정.
3. 같은 검사로 고정98의 탐색 정책과 비무작위 adaptive pruning을 비교.

Greedy deletion, attention scoring, teacher-output matching, gradient 기반 token selection, risk calibration 자체는 기존 연구다. LeaF와 SIS를 직접 다루고, output-L2 기준과 동일 비용 비교에서 차이를 보여야 한다.

Audit는 잘 맞지만 student 성능이 개선되지 않으면 “이 proxy가 학습에 필요한 것을 찾는다”는 가설이 실패한 것이다. A가 k-matched random과 같으면 교체량 조절 이상의 기여가 부족하다. B가 계산 절약을 못 하면 효율 방법으로 주장할 수 없다. 원본 정확도만 오르면 배경 강건성 개선이라고 쓰지 않는다.

우선순위는 **방법 A의 작은 진단 및 pilot**, 이후 효과가 있으면 방법 B와 효율 selector로 확장하는 순서다. A는 현재 Random10에서 관찰한 작은 이득의 원인을 좁혀 검증할 수 있어 가장 직접적인 다음 실험이다.
