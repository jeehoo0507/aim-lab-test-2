# 실험 2 최종 실행 설계

작성일: 2026-09-22. 상태: 구현·CPU 검증, 실제 데이터 준비 및 서버 학습 미실행.

**목표.** Student-guided masking에 의한 teacher 정보 제한이 일시적인 curriculum을 넘어 학습 후반에도 남는지, 이를 바꾸는 방법이 student의 오류 교정과 일반화 성능까지 개선하는지 검증한다. "Teacher가 덜 정확하면 증류가 나쁘다"를 전제로 삼지 않는다.

**논문과의 구분.** 원 논문 §5.1은 초기 teacher 정확도 저하와 빠른 student 학습을 curriculum으로 해석한다. Fig.5는 초기 전경 동시 제거·주변부 선택을 설명하고, Table5는 후반의 DINO+random을 다룬다. 따라서 초기 잘못된 선택의 발견 또는 random 추가만으로 새로운 가설이 입증되지는 않는다. [논문 §5.1](https://arxiv.org/html/2302.10494v4#S5.SS1)

**1. 데이터.** COCO instances의 모든 주석 category가 한 종류인 이미지를 고른다. 같은 class 객체 여러 개는 허용하고 mask는 합집합이다. 클래스는 giraffe, airplane, clock, zebra, train, bird, elephant, toilet, cow, bear. Crowd·비정상 면적·segmentation 누락 후보는 제외한다. 한 이미지의 다른 class 주석을 지워서 단일 class로 만드는 방식이 아니다.

| 항목 | 설계 |
| --- | --- |
| 주석상 후보 | train2017 12,370 / val2017 533; 앞선 공식 주석 집계 결과 |
| 학습 | train2017에서 class당 600, 목표 6,000 |
| 검증 | train2017에서 class당 100, 목표 1,000 |
| 고정 probe | 위 검증 중 class당 20, 총 200; 모든 seed 동일 ID |
| 최종 test | val2017의 품질 검사 후 잔여 이미지, 최대 후보 533 |
| 품질 검사 | 이미지 decode·크기·mask 유효성, decoded pixel hash 중복 제거, dHash 거리 ≤4 보수적 근접 중복 제거 |
| 고정 분할 | seed 20260922, JSON ID·파일 checksum 저장, 재실행 시 그대로 검증·재사용 |

QC 후 부족하면 class별 공통 train/val 수량을 동일하게 낮춘다. probe 20/class를 확보하지 못하면 실패로 보고하고 새 프로토콜을 작성한다. 결과가 잘 나오는 class를 골라 갈아끼우지 않는다. dHash는 near-duplicate 휴리스틱이며 의미적으로 비슷한 모든 이미지를 식별한다고 보장하지 않는다.

이미지 전체를 224×224로 resize하고 학습 시 좌우 반전만 사용한다. ImageNet normalization. Mask는 nearest resize와 동일 반전을 적용한다. 첫 버전은 crop으로 정답 객체가 사라지는 문제를 피하기 위해 random crop을 쓰지 않는다. 비율 왜곡과 단순 증강은 본 subset 프로토콜의 한계로 기록한다. 객체가 16×16 patch의 50% 이상을 차지하면 전경 patch로 정의하고, threshold에 덜 의존하는 pixel-coverage 기반 누락률도 함께 저장한다. 전경 patch 0개인 이미지는 누락률을 NA로 기록하며 분류 평가에서는 유지한다.

COCO는 자연 이미지지만 배경 상관관계가 사라진 데이터셋은 아니다. 해당 subset 결과를 COCO 표준 detection benchmark 성능으로 표기하지 않는다. Task는 10-way single-label classification이고 YOLO detector 학습으로 바꾸지 않는다. [COCO 공식](https://cocodataset.org/)

**2. 모델과 공정한 비교.** Teacher DeiT-S(10-way head 21,669,514 parameters), Student DeiT-Tiny(5,526,346). Teacher ImageNet pretrained → 30 epochs fine-tuning → validation macro best를 freeze. Seed 0/1/2마다 teacher 하나를 학습하며 두 student 초기화와 모든 방법이 같은 seed의 teacher를 공유한다. 따라서 seed 간 변동은 student뿐 아니라 teacher 차이도 포함한다.

Student는 scratch / ImageNet pretrained를 따로 100 epochs 학습한다. 같은 seed·초기화의 7방법은 동일 초기 state, minibatch 순서, 좌우 반전·drop-path RNG를 사용한다. 마스크 RNG는 별도 generator로 분리한다. 초기 tensor checksum과 teacher checkpoint checksum을 저장한다. Pretrained는 head만 새로 초기화하고 전체 backbone을 fine-tune한다.

**3. 7가지 방법.** Student는 항상 full 196개 spatial patch와 CLS를 본다. Top-k는 last block의 head-mean CLS→patch attention이다. Teacher는 positional embedding을 더한 뒤 spatial token을 gather하며 CLS는 유지한다.

| 방법/코드명 | Teacher 입력 |
| --- | --- |
| CE / `ce` | 학습에서 teacher 없음; probe용 teacher 출력은 별도 측정 |
| Full KD / `full` | 196개 |
| Random Mask / `random` | 매 step uniform without-replacement 98개 |
| MaskedKD / `student` | student top-98 |
| Random Rescue 10 / `random_rescue_10` | top-98 중 무작위 10개 제거, 원래 선택 밖 98개 중 무작위 10개 추가 |
| FG Rescue 10 / `foreground_rescue_10` | 선택된 BG와 누락된 FG 중 균등 선택하여 최대 10개 교체 |
| Low-score Rescue 10 / `low_score_rescue_10` | top-98에서 attention이 가장 낮은 10개를 제거, 선택 밖에서 무작위 10개 추가 |

모든 masked teacher는 98개를 유지한다. Low-score 방식의 보존된 88개가 정답 패치라는 보장은 없고, 기존 편향도 보존할 수 있다. 비교 결과를 보고 우열을 판단한다. FG는 segmentation을 사용한 **진단용 oracle**이며 학습 성능의 이론적 상한이나 실용적으로 공정한 경쟁 방법은 아니다.

이번 random/low-score 방식은 FG GT 없이 정확히 10개를 교체한다. 이전 Waterbirds random rescue는 FG 가능 수에 맞춘 최대10 교체였다. 이 차이를 명시한다. 고정-checkpoint probe에는 FG와 교체 수를 맞춘 `random_matched` 대조도 포함한다. FG의 가변 교체 수와 고정10 training 결과 차이를 전경 정보만의 효과로 해석하지 않는다.

**4. 학습.** AdamW, weight decay .05, label smoothing .1, student/teacher drop-path .1, grad clip 1, AMP(CUDA). Micro batch32, gradient accumulation4, 유효 batch128. Scratch peak LR5e-4, pretrained/teacher5e-5, warmup5 + cosine floor1e-6. 숫자는 실제 LR로 추가 scaling하지 않는다. KD는 `.5*CE + .5*KL(T_masked || Student)`, temperature1; CE-only loss는 CE. [공식 실행 예시](https://github.com/effl-lab/MaskedKD)

Seed0 CE·FullKD pilot을 먼저 돌려 loss/val 학습 정상 여부를 확인한다. LR은 pilot 시작값으로, 필요하면 두 baseline의 validation만으로 수정하고 프로토콜을 고정한다. 수정 시 output root와 config를 새로 만들고 모든 방법을 동일하게 다시 돌린다. 방법마다 잘 나온 LR을 선택하지 않는다. 최종 test는 이 과정에서 열지 않는다.

**5. 주 분석.** Pretrained / scratch를 섞어 평균 내지 않는다. Primary contrast는 Low-score10 vs MaskedKD의 test macro accuracy, secondary는 학습 속도·오류 교정·FG/Random/Full 비교다. Best validation macro checkpoint(동률은 이른 epoch)와 last100 결과를 모두 보고한다. WGA 대신 overall/macro/per-class accuracy를 사용한다.

- Teacher 제한: KL(T_full || T_actual), full 정답→masked 오답 비율. Raw top98 입력도 함께 기록.
- Student 선택: 교체 전 BSR/FMR, 입력 전체 BG 비율 대비 초과 선택, pixel coverage 기반 전경 면적 누락. Teacher 입력인 교체 후 값과 별도 표시.
- 모방: KL(T_full || Student), teacher-student prediction agreement. Teacher를 따라도 teacher 오답을 함께 모방할 수 있으므로 정답 정확도를 우선한다.
- 오류 교정: epoch0에서 teacher가 맞고 student가 틀린 고정 probe 집합. 3개 연속 epoch 정답이 된 첫 시점, 25/50/end까지 교정된 비율, 끝까지 실패한 표본의 검열(censoring)을 포함한 대기 시간.
- 학습 속도: validation macro curve, 80%를 3 epochs 연속 넘는 첫 step/학습 시간(미도달은 미도달), 학습 시간과 probe 비용을 분리. 추가 목표 정확도는 pilot 종료 전에 고정한다.
- 랜덤 분석: 교체 시 실제 들어오고 나간 FG 수, teacher 정답/오답 전환. 진단 시 이미지별 독립 random stream을 epoch/seed 간 고정하고 정해진 시점에서 5회 반복.

같은 이미지의 FG 총수 F, kept K=98이면 BSR=1−F(1−FMR)/K다. 둘은 완전히 독립적인 두 증거가 아니다. Attention 일치도가 낮거나 몸통/부리 선택이 다르다는 이유만으로 잘못된 증류로 판정하지 않는다.

**6. 가설 판단.** 선택률 개선만 있으면 선택 변화. Teacher 손상 회복만 있으면 teacher 입력/출력의 변화. 여기에 student 오류 교정 가속·validation 및 독립 test 개선이 반복되면 **이 조건에서 정보 제한이 학습 병목으로 작용했다는 가설을 지지**한다. Mask policy 개입 효과는 평가할 수 있지만 효과가 attention 이동만을 통해 발생했다는 mediation/인과 주장은 따로 검증해야 한다.

새 방법이 개선하지 않으면 실패도 보고한다. 특히 초기 teacher 정확도가 낮아도 student가 더 빨리 학습하면 논문의 curriculum 설명과 일관되는 결과다. 후반 KL이 크다는 이유만으로 병목이라고 부르지 않는다. COCO 한 subset과 한 모델 pair로 모든 자연 이미지/task에서 성립한다고 일반화하지 않는다.

**7. Epoch·모델 크기·통계.** 100 epochs는 비교 예산이며 수렴 보장이 아니다. 마지막20 epochs의 train/val 곡선을 확인하고 아직 개선 중이면 해당 초기화의 비교 방법을 모두 동일한 연장 schedule로 비교한다. 연장 실험은 원래100결과와 구분하며 특정 승자만 연장하지 않는다. Scratch가 모두 낮고 pretrained는 정상이라면 데이터 크기/초기 특징 학습 문제가 크다는 단서다. Teacher 크기는 이번 기본 실험에서 고정하므로 크기 원인까지 배제하지 못한다.

Seed3개 평균±SD와 paired seed 차이·95% t interval을 보고한다. 200 probe 이미지, 여러 epoch, 5개 random mask는 추가 독립 seed가 아니다. 3시드와 최대533 test 이미지만으로 작은 차이의 유의성을 확정하지 않는다. 강한 성능 우위 주장을 하려면 주요 비교와 추가 seed 수를 미리 정한 확인 실험이 필요하다. 유의해질 때까지 seed를 하나씩 추가하지 않는다.

**8. 후속 시간대 개입(현재42회와 별도).** 원 논문의 초기/후반 해석과 직접 구분하려면 MaskedKD1–50 공통 checkpoint에서 51–100을 MaskedKD 유지/FG10/Random10/Low-score10으로 분기해야 한다. 동일 weights·optimizer·schedule에서 시작해야 하므로 MaskedKD의 epoch50 전체 상태를 저장한다. 이 분기 실행기와 early-only ablation은 현재 기본42회에 포함되지 않으며, 저장만으로 해당 인과 실험을 수행했다고 주장하지 않는다.

Fig.4처럼 학습 전체를 합친 선택 빈도 그림만으로 attention 이동의 지연을 판별할 수 없다. 추가 시각화는 동일 이미지의 early(1–20) / middle(41–60) / late(81–100) 지도, patch 첫 선택 epoch·연속 누락 기간, 복구 전후 teacher의 정오를 함께 표시한다. 이 분석에 필요한 raw attention/indices는 매 epoch 저장한다. GT 객체 영역을 정답에 필수적인 패치와 동일시하지 않으며, 나뭇가지 등 문맥 영역의 선택 자체를 오류로 단정하지 않는다.

**9. 비용·재개.** 기본 jobs1, 최대 student 조건 7개 병렬. 각 프로세스 CPU threads2/workers2. Seed별 teacher 3개를 먼저 최대3개 병렬로 완료한 뒤, 42개 student 조건을 seed와 초기화 경계 없이 공용 큐에서 실행한다. 같은 seed의 조건들은 동일 teacher checkpoint와 seed로 재현한 동일 초기 가중치를 쓰지만 출력 경로와 optimizer는 독립이다. `benchmark`는 실제 batch32·accumulation4에서 대표 student 조건을 2~7개 병렬 측정하고, 전체 작업 완료 시간이 10% 이상 줄어드는 가장 빠른 수를 권장한다. `--jobs auto`가 최신 결과를 적용하며 실행 직전 메모리가 부족해졌으면 작업 수를 낮춘다. 완료 run 재사용, 미완료 run last 상태 재개. Config/데이터/teacher/core code fingerprint 불일치 시 혼합을 막고 새 output root를 요구한다. 완료 checkpoint는 약3.8GB로 추정한다. 출력 권장 계획값5GB, 경고 기준10GB이며 데이터·환경/cache는 별도다.
