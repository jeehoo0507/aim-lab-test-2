# 기존 A5000 서버에서 제거 방식 실험 실행

이 실행기는 실험 2 저장소의 기존 서버 구성을 사용한다. 근거는 `reports/20260924_204055_692914/benchmark.json`의 NVIDIA RTX A5000 약24GB 기록이다. 서버에 직접 접속해 현재 상태를 확인한 것은 아니므로 `check`와 `benchmark`를 실제 서버에서 실행한다.

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

로컬5080의70–100시간 예상치를 서버 ETA로 쓰지 않는다. 서버는 인위적 휴식이 없지만 GPU와 CPU가 다르고, 특히 deterministic_audit의 반복 검사 비용은 여전히 크다. `outputs/deletion_server/benchmark.json`과 첫 실제 epoch의 `history.json` 시간을 기준으로 판단한다. 기존 Random10과 같은 비용이라고 가정하지 않는다.
