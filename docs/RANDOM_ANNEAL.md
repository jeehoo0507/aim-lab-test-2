# 작은 후속 실험: Random 10→0 vs MaskedKD

두 방법만 비교한다. **줄이는 것은 랜덤 교체 개수이며 teacher 입력은 항상 98패치(+CLS)**다. Student는 계속 전체 196패치를 본다. `random` 전체 98개 무작위 선택과는 다르다.

| 조건 | 1–20 epoch | 21–79 epoch | 80–100 epoch |
|---|---|---|---|
| MaskedKD (`student`) | student top-98 | student top-98 | student top-98 |
| Random 10→0 (`random_anneal_10`) | top-98 중 랜덤 10개 제거, 선택 밖 랜덤 10개 추가 | 교체 개수를 선형으로 감소 | 교체 0개 = student top-98 |

정수 교체 개수는 `max(0, min(10, ceil(10 * (80 - epoch) / 60)))`. Epoch 20에는 10개, 50에는 5개, 80에는 0개다. 0개일 때는 같은 모델의 MaskedKD 선택과 정확히 같다. 높은 attention도 제거할 수 있는 기존 Random Rescue 규칙을 유지한다. GT 전경 마스크는 교체 결정에 사용하지 않는다.

기본은 **seed 0 × Scratch/사전학습 × 두 방법 = student 4회**다. DeiT-S teacher는 기존 실험 2의 해당 seed `best.pt`를 복사해 고정하며 재학습하지 않는다. Teacher 데이터 hash·클래스 수·모델 크기·seed를 확인한다. 같은 초기화 안에서 student 초기 가중치, teacher, 데이터 순서·증강, LR, KD loss, 100 epoch 예산을 맞춘다. Mask 난수는 데이터/증강 난수와 분리된다. 교체 수를 줄일 때 optimizer와 LR을 재시작하지 않는다.

이 일정은 후속 pilot을 위한 사전 고정안이지 최적 시점이 아니다. 기존 test를 이미 확인했으므로 이 평가도 탐색적 재사용 평가다. 주 지표는 epoch 100 test macro accuracy, 보조는 validation-best의 test와 학습/전경 선택/teacher KL 곡선이다. 두 결과를 모두 보고한다. Seed 0만으로 유의성을 주장하지 않는다. **이 비교만으로 점진 감소가 Random 10 유지보다 낫다고 결론 낼 수는 없다.** 우선 MaskedKD 대비 변화만 확인한다.

## 서버 실행

실험 2 저장소에서 실행한다. 기존 Git 인증·데이터·가상환경을 재사용한다.

```bash
git pull --ff-only origin main
mkdir -p logs
nohup bash setup.sh anneal --jobs 4 > logs/anneal_launcher.log 2>&1 < /dev/null &
```

기본은 새 `outputs/random_anneal/`에서 4개를 병렬 학습하고, 전부 완료되면 best/last test 평가와 분석을 순서대로 실행한다. 기존 `outputs/experiment2/` student 결과를 수정하지 않는다. 본 실험 때 7개 병렬을 사용했어도 현재 GPU에 다른 작업이 있으면 `--jobs 2`로 줄일 수 있다. DataLoader workers는 0이다.

```bash
tail -f logs/anneal_launcher.log
```

개별 epoch 진행은 다음으로 본다. `Ctrl+C`는 이 로그 보기만 종료한다.

```bash
tail -n 2 -f outputs/random_anneal/_jobs/seed_0_*.log
```

마지막 `Complete comparison: .../analysis/comparison.md`가 출력되면 두 초기화의 학습·test 평가·그림 저장까지 완료된 것이다. 프로세스 실패 시 launcher와 해당 개별 로그를 확인한다. 같은 명령 재실행 시 완료 run은 재사용하고 중단 run은 마지막 완료 epoch부터 재개한다. 같은 출력 경로를 동시에 두 번 실행하면 잠금 오류로 중복 실행을 막는다.

```bash
# 완료 후 요약 확인
cat outputs/random_anneal/analysis/comparison.md

# 분석 자료만 새 reports/<timestamp>/에 저장하고 GitHub에 업로드
bash setup.sh anneal-export --push
```

서버에서의 결과 push가 `fetch first`로 거절되면 생성된 보고서 commit은 로컬에 남아 있다. 새 보고서를 또 만들거나 amend/force-push하지 않고 기존 commit을 원격 변경과 합친 후 push한다.

```bash
git pull --rebase origin main
git push origin main
```

충돌이 있으면 충돌 내용을 먼저 확인한다. 자동으로 파일을 버리지 않는다.

다른 저장 경로를 사용했다면 `--teacher-root /기존/outputs/experiment2 --data-root /데이터/coco_single`을 `anneal` 명령에 추가한다. `--output-root`를 바꿨으면 `anneal-export`에도 같은 경로를 준다. Scratch만 먼저 보려면 `--inits scratch --jobs 2`를 사용한다. 추가 시드는 `--seeds 0 1 2`로 명시할 수 있고 해당 seed의 기존 teacher가 모두 필요하다.

## 생성 자료

- `analysis/comparison.md`, `comparison.csv`: 두 조건의 val best/last, test best/last, best epoch.
- `analysis/anneal_comparison.png` / `.pdf`: 실제 교체 개수, 검증 학습 곡선, 최종 test 비교.
- `analysis/main_scratch.png`, `main_imagenet.png`: 교체 전 student 배경 선택·전경 누락, 실제 teacher 입력의 KL.
- `analysis/paired_seed_differences.csv`: 같은 seed에서 점진 감소 − MaskedKD 차이. 1 seed에는 유의성 구간을 제공하지 않는다.
- `comparison_protocol.json`: 전체 교체 일정, teacher 원본 경로/hash/epoch, 실행 조건.
- 매 epoch: 학습/검증 지표, validation logits, probe 200장 attention·indices·teacher/student 예측.
- 가중치: 10 epoch마다 + best/last. 진행 중 last에는 optimizer/scaler도 포함하고 완료 후 축소한다.
- GitHub export: 지표·그림·압축된 probe 분석·test logits·가중치 checksum. 가중치와 데이터는 서버에 남는다.

기본 4개 student의 보존 가중치는 약 1.0GB에 MaskedKD 50 epoch 재개 상태와 teacher 복사본이 추가된다. Probe·분석까지 포함한 출력은 대략 수 GB를 계획하되 실제 크기는 실행 후 `storage.json`으로 확인한다. 데이터·환경·기존 실험 결과는 별도다. 100 epoch를 4개 병렬로 한 번 실행하는 규모라 기존 42개 전체보다 작다. 실제 시간은 현재 서버 경합에 따라 달라지므로 첫 5 epoch의 시간으로 갱신한다.
