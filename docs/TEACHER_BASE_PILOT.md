# DeiT-Base teacher: Full KD / MaskedKD / Random10

기존 `codex/random-low-sweep`의 `c1cbe580414f004b0afbd5ede97017a7f525fde2` 결과를 기준으로 teacher 크기의 효과를 비교한다.
기본값은 **seed 0, scratch DeiT-Tiny student 3개 병렬**, 각각 100 epoch다.
Teacher는 공식 ImageNet pretrained **DeiT-Base patch16/224 (distilled 모델 아님)**를 같은 COCO 데이터에서 30 epoch fine-tuning하고 validation macro best를 고정한다.
Small teacher를 재사용하거나 Base teacher를 세 개 동시에 학습하는 실행이 아니다.

| 조건 | Teacher 패치 | Student 패치 |
| --- | ---: | ---: |
| Full KD (`full`) | 196 | 196 |
| MaskedKD (`student`) | student attention top-98 | 196 |
| Random10 (`random_rescue_10`) | top-98 중 임의 10개를 선택 밖 임의 10개로 교체 | 196 |

기존 batch32 × accumulation4, AMP, LR, 증강, 200장 validation probe와 상세 진단 5회 반복을 유지한다.
Teacher 크기는 `teacher_variant`로 선택하며 기존 config의 기본값은 `small`이다.
공식 Base 구조는 width768/depth12/heads12이고 10-class head 포함 85,806,346 parameters다. Student 구조와 초기화는 바꾸지 않는다.

## 서버 실행

이미 COCO 데이터가 준비된 **실험 2 저장소 폴더**에서:

```bash
git fetch origin codex/teacher-base-pilot
git switch codex/teacher-base-pilot
mkdir -p logs
nohup bash setup.sh teacher-base > logs/teacher_base_launcher.log 2>&1 < /dev/null &
tail -f logs/teacher_base_launcher.log
```

한 명령이 순서대로 수행하는 작업:

1. 데이터 checksum과 고정 Small 보고서의 설정·student 초기화 일치 확인.
2. 임시 Base 모델로 teacher 및 세 방법 단독 벤치마크, 이후 **Full KD + MaskedKD + Random10 실제 세 프로세스 동시 벤치마크**.
3. 측정 VRAM에 프로세스당 0.75 GiB와 GPU 전체 10%(최소 2 GiB) 여유를 더해 확인. 부족하면 다음 단계를 시작하지 않음.
4. Base teacher 30 epoch 학습 후, 고정한 teacher로 세 student를 100 epoch 병렬 학습.
5. Validation/probe 분석 저장. Test 평가는 아래 별도 명령으로 수행.

벤치마크는 준비된 실제 이미지, batch32, accumulation4, warmup2 + 측정8 optimizer update, validation/probe/checkpoint I/O를 포함한다.
임시 가중치만 쓰고 실제 학습 결과를 변경하지 않는다. Small에서 측정한 예전 VRAM 예산은 재사용하지 않는다.
기본 실행은 매번 짧은 벤치마크를 새로 수행하며, 학습은 일치하는 완료 결과를 재사용하고 중단된 epoch부터 재개한다.
진행 로그는 `outputs/teacher_base_pilot/_jobs/seed_0_student_<method>.log`에 있다.

**벤치마크만** 실행:

```bash
bash setup.sh teacher-base-benchmark
cat outputs/teacher_base_pilot/benchmark.json
```

기존 A5000 24GB에서 사전 계획은 student 3개 합계 약 8–12 GiB, teacher 준비 포함 약 5–8시간이었으나 **Base 실측값이 아니다**.
실행기가 측정된 VRAM 요구량과 시간 추정 범위를 출력한다. 짧은 벤치마크는 장시간 GPU/CPU/I/O 경합을 완전히 반영하지 않으므로 실제 epoch 시간도 확인한다.
부족한 VRAM은 다른 작업 종료 후 재시도하며, 프로세스 수나 batch를 몰래 줄여 프로토콜을 바꾸지 않는다.

완료 후 평가 및 공유:

```bash
bash setup.sh teacher-base-evaluate
cat outputs/teacher_base_pilot/analysis/teacher_size_comparison.csv
cat outputs/teacher_base_pilot/analysis/teacher_size_interaction.csv
bash setup.sh teacher-base-export --push
```

`export --push`는 새 보고서만 commit/push한다. 가중치와 원본 이미지는 서버에 남는다.
Small 비교에는 저장소에 포함된 `reports/20260923_152152_732466`의 세 방법을 사용하므로 기존 Small 가중치가 없어도 된다.
설정·데이터·초기 가중치·seed·완료 epoch가 다르면 비교를 중단한다.
CPU/PyTorch 실행 환경이 달라 초기 가중치 checksum이 달라지는 경우도 중단하므로 기존 실험을 수행한 서버 환경을 사용한다.
다른 Small 보고서나 원본 출력 폴더를 쓰려면 모든 명령에 동일한 `--small-root <경로>`를 붙인다.

기본 출력은 `outputs/teacher_base_pilot`이며 기존 `outputs/experiment2` 결과를 변경하지 않는다.
`--data-root`, `--output-root`, `--device cuda:0`을 지원한다. 실행·평가·내보내기 모두 같은 값을 사용한다.
처음부터 3 seed를 실행하려면 각 명령에 `--seeds 0 1 2`를 붙인다. Teacher는 순서대로, student는 seed별 세 방법씩 총 세 차례 실행한다.
이미 시작한 seed 0 전용 출력에 seed를 추가하지 말고 `--output-root outputs/teacher_base_3seeds`처럼 별도 폴더를 쓴다.

## 해석

주 비교는 epoch100 test macro이며 validation-best는 별도 열이다.
`teacher_size_interaction.csv`는 각 크기의 `Random10 − MaskedKD`와 그 차이의 변화를 percentage point로 저장한다.
Teacher 크기를 제외한 조건을 맞추지만, 더 큰 teacher의 효과를 attention 정확도만의 효과로 단정하지 않는다.
이미 관찰한 test set을 쓰는 탐색적 후속 실험이며 seed 0 한 번으로 일반적인 우열을 확정하지 않는다.

CPU 자동 검증은 작은 합성 모델로 실제 세 프로세스 실행·teacher 재개·평가·내보내기를 확인한다.
별도로 실제 Base 구조와 196/98-patch forward를 검증하지만 A5000 성능 수치는 서버 벤치마크가 필요하다.
