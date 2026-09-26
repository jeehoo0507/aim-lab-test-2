# 현재 PC에서 제거 방식 실험 실행하기

2026-09-26 구성. **실제 COCO 학습은 아직 시작하지 않았다.** 실행 파일은 `setup_deletion.sh`, 설정은 `configs/deletion_5080.json`이다. 실험 1 저장소의 실행기를 사용하지 않는다.

추가 갱신: 서버 병렬 실행을 준비하면서 중복 후보 탐색을 같은 결과의 배열 연산으로 최적화했다. 이후8 microbatch 합성 재측정에서 teacher30회 약0.51시간, MaskedKD100회 약0.53시간, Random10100회 약0.49시간, rescue_audit100회 약2.18시간, deterministic_audit100회 약12.78시간으로 환산됐다(합계 약16.49시간, I/O·validation 제외). 아래에 남긴 최초 약65시간 환산은 **최적화 이전 기록**이며 현재 ETA가 아니다. 실제 전체 학습은 여전히 미측정이고 서버 병렬 시간은 서버에서 다시 측정한다.

## 확인한 사양과 환경

| 항목 | 실제 조회 결과 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 총 15.92 GiB, 조회 당시 여유 약 13 GiB |
| Driver | 595.91.07 |
| CPU | AMD Ryzen 7 9800X3D, 8코어/16스레드 |
| RAM | OS 가용 총량 59.49 GiB, 조회 당시 available 약 41 GiB |
| 디스크 여유 | 별도 환경 설치 후 약 58.5 GiB |
| 새 환경 | `.venv-deletion`, Python 3.11, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128 |
| 원래 환경 | `.venv`, PyTorch 2.5.1+cu124 유지 |

RTX 5080의 Blackwell 지원을 위해 별도 CUDA 12.8 환경을 설치했다. 공식 근거: [PyTorch 2.7의 Blackwell 지원](https://pytorch.org/blog/pytorch-2-7/), [공식 이전 버전 설치 명령](https://pytorch.org/get-started/previous-versions/). 설치된 binary의 `sm_120` 포함 여부와 작은 CUDA FP16 행렬곱·convolution 동작을 확인했다. 이후 사용자 요청에 따라 임시 모델로 방법별8 microbatch GPU benchmark를 완료했다. 실제 COCO 학습은 시작하지 않았다.

## 기본 자원 설정

- **동시 학습 1개.** MaskedKD, Random10, 새 두 방법을 순차 실행한다.
- Microbatch 8 × accumulation 16 = 유효 batch 128. 평가 batch도8이다.
- CPU thread2, DataLoader worker0, 현재 프로세스 nice10.
- PyTorch allocator 한도 최대8 GiB. 시작 시 여유가 적으면 desktop 여유3 GiB를 남기도록 더 낮춘다. 이는 GPU 전체 사용량이나 다른 프로그램을 제한하는 기능이 아니다.
- 전체 GPU 여유3 GiB, RAM available12 GiB, 디스크 여유15 GiB 미만이면 중단한다.
- GPU78°C 이상 또는 GPU 온도 조회 실패 시 중단한다. 드라이버·팬·전압·GPU power limit은 변경하지 않는다.
- 계산 사이 휴식을 넣는 duty0.65. GPU utilization이나 순간 소비전력을65%로 강제 고정하는 기능은 아니다.
- 명령 한 번의 세션 한도120분. 시간·온도·메모리 중단 시 마지막 **완료 epoch**를 보존한다. 아직 완료되지 않은 epoch는 다음 실행에서 처음부터 반복한다.
- 삭제 검사는 이미지당 teacher 입력 평가 최대24개, 한 번에 후보3개까지만 묶는다. 한도에 도달하면 검증을 통과한 현재 마스크에서 멈춘다.

GPU 온도 등은 최대5초 간격으로 읽고, teacher candidate와 microbatch 경계에서 확인한다. 한 개의 실행 중인 GPU 연산을 중간에 강제로 끊는 장치는 아니다. 시작 후 첫 benchmark로 실제 메모리·속도를 확인해야 한다.

## 구현한 방법

| CLI 이름 | 동작 |
| --- | --- |
| `student` | 기존 student attention top98 MaskedKD |
| `random_rescue_10` | 기존 무작위10개 제거·추가 |
| `rescue_audit` | 추가 후보를 포함한108개 reference로 삭제를 검사. 실패하면5개 교체, 다시 실패하면 기존98개로 복귀 |
| `deterministic_audit` | 무작위 없이196개에서 시작, 최초196개 reference의 가상 학습효과를 유지하는 만큼만 제거. 목표 하한98 |
| `full` | 선택적으로 실행할 Full KD 대조군 |

새 두 방법은 [연구 설계](DELETION_METHOD_PROPOSAL.md)의 정확한 가상 logit update 검사를 사용한다. 3개 step size에서 reference loss 감소량의95% 이상을 보존하는 조건이다. 실제 학습 성능 보장을 뜻하지 않는다. `rescue_audit`의 fallback은108개 대비 검사를 통과한 압축으로 집계하지 않는다.

Teacher는 DeiT-Base, student는 DeiT-Tiny scratch, seed0, teacher30/student100 epoch이다. 초기 pilot도100 epoch LR schedule의 첫2 epoch만 실행하며, 나중에 그대로 이어간다. 이 PC에서는 런타임과 microbatch가 바뀌므로 과거 서버 결과를 새 실행의 완전히 동일한 대조군으로 취급하지 않는다. 같은 새 환경에서 baseline을 함께 돌린다.

## 실행 순서

아래 명령은 이 폴더에서 하나씩 실행하고 결과를 확인한다. 기존 `bash setup.sh`는 원래 실험 환경을 사용하므로 이번에는 `setup_deletion.sh`를 쓴다.

```bash
cd /home/cjh/문서/ChatGPT/aim-lab/aim-lab-test-2
bash setup_deletion.sh check
```

지금의 check에는 **COCO 이미지·마스크 부재, 학습된 teacher 부재**가 표시되는 것이 정상이다. 보고서 파일에 예측 결과와 dataset manifest는 있지만 이미지와 `.pt` 가중치는 없다. 설치 단계는 이미 완료했으며, 다른 컴퓨터에서 새로 구성할 때만 `bash setup_deletion.sh env`가 필요하다.

먼저 실제 GPU 메모리·대략적인 속도를 짧게 측정한다. 무작위 입력과 임시 모델로 forward/backward를 실행하며, 연구용 checkpoint를 만들거나 기존 모델을 학습시키지 않는다.

```bash
bash setup_deletion.sh benchmark
```

`outputs/deletion_5080/benchmark.json`의 각 방법별 peak VRAM과 시간 지표를 확인한다. 무작위 입력의 짧은 측정이므로 데이터 읽기·validation·학습 후반의 다른 삭제 통과율까지 포함한 확정 ETA가 아니다. 실제 학습 첫 epoch 시간으로 다시 판단한다. Benchmark의 설정·코드·런타임이 바뀌면 학습 전에 다시 측정해야 한다.

2026-09-26 `benchmark --steps 8` 측정값을 현재 epoch 수에 곱하면 teacher30회 약0.79시간, MaskedKD100회 약1.34시간, Random10100회 약1.31시간, rescue_audit100회 약5.41시간, deterministic_audit100회 약56.15시간이다. 합계는 **학습 부분만 약65시간**이다. 데이터 I/O·validation·삭제 통과율 변화와 중단된 epoch 재실행을 고려한 초기 작업 예산은70–100시간 정도로 잡되, 이는 보장 범위나 통계적 신뢰구간이 아니다. 다운로드와 수동 재시작 대기 시간은 별도다. Teacher가 준비되어 있을 때 네 방법의2 epoch pilot은 학습 부분만 약77분이다. 환산표는 `outputs/deletion_5080/time_estimate.json`에 저장했다.

짧은 측정의 PyTorch peak reserved는 teacher 약2.11GiB, 학생 방법 약0.66–0.68GiB였다. CUDA context·desktop 메모리를 포함한 GPU 전체 사용량과는 다르며, 장시간 온도나 학습 후반의 동작을 검증한 결과는 아니다. 현재120분 세션 제한 때문에 무인 연속 실행은 하지 않으며 같은 명령으로 재시작해야 한다.

기존 실험과 **동일한7,533개 image ID와 split**의 데이터를 준비한다. 전체 COCO 이미지 archive를 받지 않고, annotation archive와 지정 이미지들만 내려받는다. 원본 dataset manifest와 이미지·마스크 checksum까지 검증한다. 다운로드는2개씩 처리한다.

```bash
bash setup_deletion.sh prepare
```

학습된 DeiT-Base teacher를 이 PC에서 새로 만들려면 다음을 실행한다. **이 명령부터 실제 학습이다.**

```bash
bash setup_deletion.sh teacher
```

120분 제한으로 중단됐다면 같은 명령을 다시 실행해 이어간다. 완료 전에 student 실행으로 넘어가지 않는다. 기존 서버의 teacher를 가져왔다면 teacher 학습 대신 `--teacher-checkpoint`로 지정할 수 있다. Student 명령마다 같은 경로를 전달해야 하며, dataset hash·seed·teacher size가 맞아야 한다. 실험1의 가중치는 거부된다.

네 방법을 각각 첫2 epoch만 순차 실행한다.

```bash
bash setup_deletion.sh pilot
```

먼저 한 방법만 확인하려면 다음처럼 제한할 수 있다.

```bash
bash setup_deletion.sh pilot --methods rescue_audit
```

로그의 `COMMITTED`, validation macro, GPU peak 및 `history.json`을 확인한 다음100 epoch까지 이어간다. 한 세션에 끝나지 않으면 같은 명령을 다시 실행한다. 완료된 방법은 다시 학습하지 않는다.

```bash
bash setup_deletion.sh run
```

각각 따로 실행하는 것도 가능하다.

```bash
bash setup_deletion.sh run --methods student random_rescue_10 rescue_audit
bash setup_deletion.sh run --methods deterministic_audit
```

모두 완료한 뒤 최종 test를 명시적으로 평가한다. 학습 도중 test 점수는 계산하지 않는다.

```bash
bash setup_deletion.sh evaluate
```

## 중단·재개와 기록

`Ctrl+C`는 중단 요청으로 처리한다. 완료되지 않은 epoch의 업데이트는 checkpoint에 반영하지 않으며, `last.pt`의 완료 epoch부터 재개한다. 중단 원인은 각 run의 `STOPPED.json`에 기록한다. 온도·메모리 등 원인이 해소된 뒤 같은 명령으로 재개한다. 자동으로 재실행하거나 다른 프로세스를 종료하지 않는다.

- 실행 로그: `logs/deletion_*.log`
- 사양과 준비 상태: `outputs/deletion_5080/preflight.json`
- 사전 측정: `outputs/deletion_5080/benchmark.json`
- Teacher: `outputs/deletion_5080/seed_0/teacher/`
- Student: `outputs/deletion_5080/seed_0/scratch/<method>/`
- `last.pt`: 미완료 run은 model·optimizer·scaler·best model·history, 완료 run은 최종 model.
- `best.pt`: validation macro 최고 checkpoint, 동률은 먼저 나온 epoch.
- `history.json`: loss, validation, 실제 남긴 token·교체 수, teacher 평가 수, fallback·예산 중단율, peak VRAM.
- `audit/epoch_*.jsonl`: 새 방법의 이미지별 교체 수와 검증 결과. `last.pt`보다 뒤의 epoch 파일이나 `.part`는 미완료 기록이다.

Source code·주요 환경 버전·dataset·teacher·학습 설정 fingerprint가 달라지면 resume를 거부한다. 알고리즘이나 batch를 바꾸려면 새 `--output-root`를 사용한다. 자원 제한을 바꾸는 경우에도 benchmark 일치 검사를 확인한다.

첫 구현은 **검사 비용을 포함해 가설을 검증하는 버전**이다. 여러 teacher forward를 쓰므로 기존 Random10보다 느릴 수 있다. 실제 사용 token 수만 보고 계산 효율이 개선됐다고 해석하지 않는다. 시간 개선용 경량 selector는 아직 구현하지 않았다.

## 검증 범위

CPU 합성 데이터로 가상 학습 기준, 학생 상태에 따른 순위 변화, joint deletion, 난수 순서 보존, 삭제 평가량 제한, 기존 forward 동일성, 자원 중단, 중단 후 재개를 검사한다. 실제 COCO 정확도·장시간 GPU 메모리·열 상태는 아직 측정하지 않았다. 전용 환경에서 다음 명령으로 코드 검증을 반복할 수 있다.

```bash
.venv-deletion/bin/python -m pytest -q
```
