# 기존 A5000 서버에서 제거 방식 실험 실행

이 실행기는 실험 2 저장소의 기존 서버 구성을 사용한다. 근거는 `reports/20260924_204055_692914/benchmark.json`의 NVIDIA RTX A5000 약24GB 기록이다. 서버에 직접 접속해 현재 상태를 확인한 것은 아니므로 `check`와 `benchmark`를 실제 서버에서 실행한다.

## 추가 seed를 포함한 권장 실행

**`parallel`은 학생 seed0·1·2를 최대3개 동시에 실행한다.** 각 seed 안에서는 MaskedKD → Random10 → rescue_audit → deterministic_audit를 순서대로 수행한다. 따라서 기본 전체 계획은 **4방법 × 3학생 seed = 12개 학습**, 동시 학생은 최대3개다. 기존 Base teacher(seed0)는 모든 학생이 공유하는 고정 가중치이며 각 worker가 GPU에 별도 로드한다. 이 실험의 seed 표준편차는 **고정 teacher 아래 학생 초기화·학습 변동성**을 나타내며, teacher까지 다시 학습한3개 독립 반복을 뜻하지 않는다.

아래의 코드 받기와 `env`를 마친 뒤 서버에서 실행한다.

```bash
mkdir -p logs
nohup bash setup_deletion_server.sh parallel --seeds 0 1 2 --max-jobs 3 \
  > logs/deletion_parallel_launcher.log 2>&1 < /dev/null &
tail -f logs/deletion_parallel_launcher.log
```

자동 절차는 다음과 같다.

1. 데이터 checksum, 재사용 teacher의 dataset·크기·class 수·teacher seed, 서버 여유 자원을 확인한다.
2. 임시 모델로1개 worker를 측정한다. 추가 실험용 checkpoint는 만들지 않는다.
3. 측정한 allocator peak의1.5배+0.5GiB 이상을 worker cap으로 잡고(최소3GiB), worker별 CUDA context 예산1GiB와 GPU 전체 여유3GiB를 별도로 남긴다. RAM은 worker당3GiB 추가 여유, CPU는 worker당2스레드와 OS용2논리 CPU를 계산한다.
4. 위 예산에 맞으면2·3 worker를 실제 동시에 측정한다. 각 방법은 warmup 뒤 barrier로 측정 시작을 맞춘다. 동시에 돌릴 때 느려지는 정도와 대기 seed까지 포함해 전체 완료 시간을 비교한다.
5. 전체 처리 시간 개선이5% 이상일 때 병렬 수를 늘린다. 측정 실패 시 더 큰 병렬 수는 시도하지 않는다. 실제 학습 중 실패·온도·메모리 문제는 다른 관리 대상 worker도 중단하며 자동 재시작하지 않는다.

GPU뿐 아니라 CPU·RAM·속도 때문에1개 또는2개가 선택될 수 있다. 이 경우 요청한3개 seed는 대기열에서 순차적으로 이어간다. `--max-jobs 1`은3개 seed를 순차 실행한다. 서버의 다른 GPU 작업은 종료하지 않는다.

출력은 단일 실행과 분리한 **`outputs/deletion_server_multi`**다. 공통 로그에는 준비·측정·wave 상태가, 실제 학습 진도는 아래 개별 로그에 기록된다.

```bash
tail -f outputs/deletion_server_multi/seed_0/worker.log
tail -f outputs/deletion_server_multi/seed_1/worker.log
tail -f outputs/deletion_server_multi/seed_2/worker.log
```

동시 실행 수와 VRAM 예산·시간 예상은 `parallel_plan.json`, 고정 teacher 및 연구 설정은 `parallel_protocol.json`에 저장된다. `parallel-benchmark`는 같은 사전 검증·측정만 하고 실제 학습을 시작하지 않는다. 각 seed를 먼저2 epoch씩 검사하려면 `parallel --seeds 0 1 2 --max-jobs 3 --until-epoch 2`를 쓰고, 이후 `--until-epoch 2`를 빼서100 epoch까지 이어간다. 코드·데이터·teacher·방법·seed 목록·주요 런타임이 바뀌면 새 출력 폴더가 필요하다.

완료한 뒤 별도로 최종 test와 평균·표준편차를 계산한다.

```bash
bash setup_deletion_server.sh parallel-evaluate --seeds 0 1 2
tar --exclude='*.pt' --exclude='audit' --exclude='*.lock' \
  -czf deletion_server_multi_results.tar.gz outputs/deletion_server_multi
```

`multi_seed_results.json`은 epoch100과 validation-best 결과를 따로 저장한다. test는 학습·병렬 수 결정에 사용하지 않는다.

## 기본값

| 항목 | 서버 설정 |
| --- | --- |
| 실행 환경 | 기존 `.venv`, `uv.lock`의 PyTorch2.5.1 CUDA12.4 |
| 데이터 | 기존 `data/coco_single`, 고정 COCO2017 단일 라벨10종 split |
| Teacher | `outputs/teacher_base_pilot/seed_0/teacher/best.pt` 재사용 |
| Student | DeiT-Tiny scratch, seed0, 각100 epoch |
| 방법 | MaskedKD → Random10 → rescue_audit → deterministic_audit 순차 실행 |
| Batch | microbatch32 × accumulation4 = 유효128 |
| CPU | thread2, DataLoader worker0, nice5 |
| GPU | PyTorch allocator 최대12GiB, GPU 전체 여유3GiB 유지 |
| 중단 기준 | GPU80°C 이상, RAM available8GiB 미만, disk15GiB 미만, GPU 상태 조회 실패 |
| 실행 시간 | **2시간 제한 없음**, 계산 사이 인위적 휴식 없음 |
| 결과 | 새 `outputs/deletion_server`에 저장 |

온도와 메모리는 계속 확인한다. 중단 조건에 걸리면 자동 반복하지 않으며, 원인을 해소한 뒤 같은 명령으로 재개한다. 드라이버·전력 제한·팬 설정은 변경하지 않는다. 네 방법은 각각 한 프로세스에서 순서대로 실행하며 기존 결과를 덮어쓰지 않는다. 두 새 방법의 알고리즘과 삭제 검사 예산은 [로컬 설계](LOCAL_DELETION_5080.md)와 같다.

## 서버에서 코드 받기

이미 사용한 서버 경로 기준이다. 폴더가 다르면 첫 줄만 바꾼다. `git switch`가 수정 파일 충돌로 실패하면 뒤의 실행 명령으로 넘어가지 않는다.

```bash
cd /home/kebap/Desktop/workspace/34/aim-lab-test-2
git fetch origin codex/deletion-server
git switch codex/deletion-server
git merge --ff-only origin/codex/deletion-server
bash setup_deletion_server.sh env
bash setup_deletion_server.sh check
```

`env`는 원래 실험의 lockfile에 맞춰 환경을 확인한다. 서버에서는 로컬5080용 `setup_deletion.sh env`를 실행할 필요가 없다. `check`는 학습하지 않고 준비 상태를 출력한다. 데이터·teacher·자원이 맞지 않으면 종료 코드2로 실패한다. 잘못된 dataset·seed·teacher 크기·class count는 거부한다. 기존 A5000을 RTX5080으로 잘못 판정하지 않는다.

## 한 명령으로 준비 확인 → 측정 → 본 학습

```bash
mkdir -p logs
nohup bash setup_deletion_server.sh start \
  > logs/deletion_server_launcher.log 2>&1 < /dev/null &
tail -f logs/deletion_server_launcher.log
```

`start`는 strict check → 서버 GPU benchmark(방법별8 microbatch) → 학생 네 방법100 epoch를 수행한다. 앞 단계가 실패하면 뒤 단계로 넘어가지 않는다. **기존 teacher를 재학습하거나 데이터를 새로 다운로드하지 않는다.** 실행 버튼 역할이므로, 실제 학습을 시작할 준비가 됐을 때만 입력한다. `tail -f`의 `Ctrl+C`는 로그 보기만 종료한다.

각 방법을 우선2 epoch만 확인하려면 다음을 사용한다. 학습률은100 epoch schedule을 유지하며 `run`으로 그대로 이어간다.

```bash
bash setup_deletion_server.sh check && \
bash setup_deletion_server.sh benchmark --steps 8 && \
bash setup_deletion_server.sh pilot
```

준비와 benchmark가 완료됐다면 `bash setup_deletion_server.sh run`으로 바로 학습을 시작하거나 이어갈 수 있다. `start` 재실행은 benchmark를 한 번 더 수행한 뒤 이어간다. 설정·코드·런타임·GPU가 바뀐 benchmark는 재사용하지 않는다.

## 경로가 다른 경우

Teacher가 다른 곳에 있다면 `check`, `start`, `pilot`, `run`, `evaluate`에 같은 경로를 전달한다. 경로에 공백이 있으면 따옴표로 감싼다.

```bash
bash setup_deletion_server.sh check --teacher-checkpoint /path/to/teacher/best.pt
bash setup_deletion_server.sh start --teacher-checkpoint /path/to/teacher/best.pt
```

데이터는 `--data-root`, 출력은 `--output-root`로 바꾼다. 관련 명령에 같은 경로를 전달한다. 기존 결과의 checkpoint hash와 설정을 기록하므로 다른 teacher를 중간에 바꾸면 resume가 거부된다.

기존 teacher가 실제로 없을 때만 별도로 `benchmark`, `teacher`를 실행해 새 teacher를 만든다. 이때 생성 경로는 `outputs/deletion_server/seed_0/teacher/best.pt`이며 이후 명령에 해당 경로를 `--teacher-checkpoint`로 전달한다. 데이터가 없을 때는 `prepare`로 기존 image ID와 checksum을 재현할 수 있다. 이 단계들은 `start`가 자동 실행하지 않는다.

## 완료와 평가

학습 로그의 `COMMITTED <method> epoch 100/100` 및 각 run의 `result.json`으로 완료를 확인한다. test는 자동 평가하지 않는다.

```bash
bash setup_deletion_server.sh evaluate
```

결과는 `outputs/deletion_server/seed_0/scratch/<method>/`의 `history.json`, `test_metrics.json`에 있다. 새 방법의 `audit/epoch_*.jsonl`에는 이미지별 남긴 토큰·교체 수·검사 비용·fallback이 저장된다. 가중치와 대용량 audit을 제외한 분석용 결과 묶음은 다음처럼 만들 수 있다.

```bash
tar --exclude='*.pt' --exclude='audit' --exclude='.deletion.lock' \
  -czf deletion_server_results.tar.gz outputs/deletion_server
```

실행 중단 시 마지막 **완료 epoch**부터 재개하고, 중단된 epoch는 다시 수행한다. `STOPPED.json`에 원인이 기록된다. 출력 경로를 공유하는 중복 실행은 lock으로 거부한다.

## 검증과 시간 해석

서버와 같은 PyTorch2.5.1 환경의 CPU 합성 테스트로 두 방식의 학습·재개, 연속 실행, 자원 중단, 재사용 teacher 검사, A5000/5080 구별, 실패한 준비 확인 뒤 학습이 시작되지 않는 동작을 확인했다. 실제 A5000 GPU benchmark와 COCO 학습은 사용자가 서버에서 시작한다.

병렬 확장 후 전체 테스트88개가 통과했다. 현재 RTX5080에서는 로컬 보호 설정을 유지한 임시 GPU worker1개·2개 calibration도 통과했고,2개가 전체 처리량 개선 조건을 만족해 선택됐다. 이는 병렬 실행기 동작 검증이며, A5000에서3개가 가능하다는 실측은 아니다.

로컬5080의70–100시간 예상치를 서버 ETA로 쓰지 않는다. 서버는 인위적 휴식이 없지만 GPU와 CPU가 다르고, 특히 deterministic_audit의 반복 검사 비용은 여전히 크다. `outputs/deletion_server/benchmark.json`과 첫 실제 epoch의 `history.json` 시간을 기준으로 판단한다. 기존 Random10과 같은 비용이라고 가정하지 않는다.

2026-09-26 추가 최적화: 중복 후보의 조건부 max 유사도 탐색을 배열 연산으로 바꿨고, 무작위·동률·영벡터 입력에서 기존 scalar 구현과 제거 순서가 같음을 검사했다. 로컬의 단일 worker 짧은 합성 측정에서 deterministic_audit는100 epoch 환산 약56시간→약13시간, 학생4방법 합계 약15.98시간으로 변했다. 로컬 설정(batch8, duty0.65)의 값이며 서버 속도나3 worker 속도로 해석하면 안 된다. A5000의 과거 학생 allocator peak는약1.6GiB였으므로 실험 프로세스1개 약2–4GiB,3개 약8–12GiB를 초기 예상으로 삼을 수 있으나, 실제 병렬 실행 허가는 항상 새 서버 측정과 여유 자원으로 결정한다. 시간의 임시 작업 예산은1 seed 전체20–40시간,3 seed 병렬 전체30–80시간 정도로 넓게 잡되, 서버 측정 전 외삽일 뿐 보장 범위가 아니다. 실제 `parallel_plan.json`과 첫 epoch 기록이 이 예상보다 우선한다.
