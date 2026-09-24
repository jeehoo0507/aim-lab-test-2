# DINO attention에서도 Random10 이점이 나타나는가

이 실험은 **MaskedKD 논문 §5.1 / Table 5의 DINO attention 선택 비교**를 COCO 실험 2에 적용한다. 기존 Base teacher를 유지하고 패치 선택에 고정된 DINO v1 ViT-S/16을 쓴다. DINO 자기지도 학습(논문 §4.3 / Table 4), DINO를 분류 teacher로 바꾸는 실험, ImageNet 원논문 수치의 재현과는 구분한다.

참고 자료:

- [MaskedKD 논문](https://arxiv.org/html/2302.10494v3): §5.1과 Table 5의 고정 DINO attention 및 후반 무작위 혼합.
- [MaskedKD 공식 코드](https://github.com/effl-lab/MaskedKD/tree/96d052da7346441e2425876a9ad37ff8c87fe383): 마지막 블록 head 평균 CLS→patch attention, top-k, teacher 입력 토큰 제거, soft KD 손실.
- [DINO 공식 코드](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a): `get_last_selfattention`, ViT-S/16 공식 가중치. 참조 코드와 라이선스는 `vendor/DINO/`에 보관한다.

## 비교 조건

세 방법 모두 동일한 seed의 **기존 COCO DeiT-Base best teacher**, scratch DeiT-Tiny 초기 가중치, 학습 이미지 순서·증강, CE/KL 비중·학습률·100 epoch를 사용한다. DINO는 ImageNet 자기지도 사전학습 가중치를 그대로 고정하고 학습하지 않는다. DINO와 student·teacher에는 동일하게 증강된 224×224 이미지를 전달한다. Student는 196개 공간 패치 전부, teacher는 98개와 CLS를 사용한다.

| 실행 이름 | epoch 1–50 | epoch 51–100 |
|---|---|---|
| `dino` | DINO attention top 98 | 동일 |
| `dino_random_rescue_10` | top 98에서 무작위 10개 제거 + 선택 밖 10개 추가 | 동일 |
| `dino_paper_late` | DINO attention top 98 | top 78 유지 + 나머지 118개 중 무작위 20개 |

Table 5의 “상위 attention 40% + 무작위 10%”를 196개 격자에 옮겨 78+20으로 정했다. 논문의 비율을 정수화한 대조군이며, 현재 Random10의 **10개 교환**과 같은 조건이 아니다. 후반 대조군의 무작위 추첨 풀은 top 78 이외 전체 118개로 정의했다. 그 안에 원래 top 98의 나머지 20개가 포함되므로, 실제 top 98 밖에서 들어온 개수는 20보다 작을 수 있다. 매 batch의 실제 교환 수를 기록한다. Foreground 정답은 세 방법의 선택에 쓰이지 않는다.

학습 시 무작위 패치는 매번 새로 뽑고, 모델·데이터 증강 RNG와 별도 generator를 쓴다. Probe는 이미지·방법·반복별 추첨을 고정하여 epoch 비교를 안정화한다. **DINO는 고정 모델이고 probe 이미지도 고정되어 있으므로, DINO top 98과 고정 추첨 Random10의 teacher probe 곡선이 평평한 것은 정상**이다. 이 probe 곡선을 학습 중 실제 마스크 다양성의 측정값으로 해석하면 안 된다.

`probe/*.npz`의 `attention`과 `raw_indices`는 이 실험에서 DINO 선택 attention과 top 98이다. 학생 자체 attention은 별도 `student_attention` 배열에 보존한다. `probe/manifest.json`의 `selection_source`에도 이를 명시한다. 기존 실험에서는 계속 학생 attention을 사용한다.

## 서버에서 실행

기존 `teacher-base` 실험을 완료한 실험 2 저장소 안에서 실행한다. 기본값은 다음을 재사용한다.

- Teacher 체크포인트: `outputs/teacher_base_pilot/seed_0/teacher/best.pt`
- 기존 Full KD·MaskedKD·Random10 비교 결과: `reports/20260924_204055_692914`
- 새 출력: `outputs/dino_attention_pilot`

초기에 데이터 manifest, teacher 체크포인트 해시, baseline 학습 설정·초기 가중치를 대조한다. DINO 공식 가중치는 한 번 다운로드하고 SHA256을 검증한 뒤 각 프로세스에서 고정 모델로 로드한다. 기존 teacher를 복사해 재사용하며 재학습하지 않는다.

```bash
# DINO forward를 포함한 단독/3개 병렬 벤치마크만 실행
bash setup.sh dino-benchmark

# 벤치마크 → GPU 여유 메모리 확인 → 학생 3개 병렬 학습
mkdir -p logs
nohup bash setup.sh dino > logs/dino_launcher.log 2>&1 < /dev/null &
tail -f logs/dino_launcher.log
```

`run`은 benchmark를 직접 포함하므로 사전 benchmark 명령을 생략해도 된다. `--jobs 1` 또는 `--jobs 2`로 동시 실행 수를 줄일 수 있다. DINO forward의 비용이 추가되므로 이전 Base 학생 벤치마크 값을 그대로 재사용하지 않는다. GPU당 프로세스의 DINO·student·teacher와 optimizer·probe를 함께 측정하고, 추가 여유 메모리를 확보한 뒤 실행한다. 보고되는 시간은 짧은 벤치마크의 외삽이며 실제 장시간 실행 측정값은 아니다.

완료 후:

```bash
bash setup.sh dino-evaluate
bash setup.sh dino-export --push
```

`run` 중에는 test를 평가하지 않는다. `evaluate`에서 세 학생의 완료 여부를 확인한 뒤 last / validation-best를 모두 평가한다. 기존 baseline의 test 지표도 이 단계에서 점수로 읽어 비교한다. `export --push`는 새 compact report만 commit/push하고 체크포인트와 전체 attention NPZ는 서버에 남긴다.

중단되면 동일 명령으로 다시 실행한다. 마지막 완료 epoch의 optimizer/scaler에서 재개하며, 데이터·teacher·DINO selector·학습 코드가 달라졌으면 기존 실행에 이어 쓰지 않는다. 이전 DeiT 학습 코드를 수정했으므로 이미 완료한 teacher-base 실행은 다시 학습시키지 말고 이 DINO 전용 진입점을 사용한다.

경로를 바꿀 때는 실행·평가·export에 같은 인자를 준다.

```bash
bash setup.sh dino --teacher-root outputs/teacher_base_pilot \
  --baseline-report reports/20260924_204055_692914 \
  --output-root outputs/dino_attention_pilot --seeds 0 --jobs 3
```

추가 seed는 해당 seed의 기존 teacher와 세 baseline이 모두 준비된 경우 `--seeds 0 1 2`로 별도 output root에서 실행할 수 있다. 하나의 등록된 protocol에서 seed 목록을 중간에 바꾸지 않는다.

## 어떤 결과를 보면 되는가

- `analysis/dino_comparison.csv`: 기존 Full KD·MaskedKD·Random10과 새로운 세 DINO 조건, best/last 각각의 test macro.
- `analysis/dino_random10_effect.csv`: 학생 attention에서 Random10의 이득, DINO attention에서의 이득, 두 이득의 차이, DINO 후반 혼합 이득.
- `analysis/dino_interpretation.json`: seed별 차이와 분석 범위.
- `analysis/learning_curves.csv`, `probe_curves.csv`, `counterfactual.csv`: 학습·teacher 예측 변화·foreground 선택 진단.
- `comparison_protocol.json`: 가중치·데이터·학습 코드·기존 비교군 해시와 조건 정의.
- `benchmark.json`: DINO 계산까지 포함한 시간·VRAM 측정 및 예상치.

일차 비교는 **동일 seed, epoch 100의 DINO+Random10 − DINO top98**이다. 기존 학생 attention에서의 Random10 이득과 나란히 보고한다. DINO에서도 양수이면 무작위 교환의 이점이 현재 학습 중인 학생 attention에만 한정되지 않을 가능성을 지지한다. 0 또는 음수이면 모든 selector로 일반화된다고 주장할 수 없다. 한 seed, 작은 test, 기존에 관찰한 test라는 제약이 있으므로 반복 seed가 필요하며 probe 반복을 학습 seed로 세면 안 된다.

후반 혼합 대조군은 시점, outgoing 선택, 무작위 패치 수가 함께 다르다. 그 우열만으로 “후반 시작” 하나의 인과효과를 분리할 수는 없다. COCO 부분집합·DeiT-Tiny·Base teacher·증강 설정도 원논문과 달라 원논문의 ImageNet 수치와 직접 비교하지 않는다.

로컬 검증은 CPU 합성 데이터의 전체 실행·재개·평가·export와 공식 구현 출력 비교다. 실제 COCO GPU 성능과 소요 시간은 서버 실행 후 판단한다.
