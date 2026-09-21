# 저장·내보내기와 가능한 분석

경로: `outputs/experiment2/seed_<s>/teacher/`, student는 `seed_<s>/<scratch|imagenet>/<method>/`.

| 파일 | 내용 | 시점 |
| --- | --- | --- |
| `config.json`, `result.json` | 설정, seed/init/method, 모델 초기 tensor checksum, teacher/data/code hash, best epoch | 시작/완료 |
| `history.json` | 실제 LR, CE/KD/loss, 실제 swap/FG 유입·유출, optimizer/AMP skip, val class별 성능, 시간/VRAM | 매 epoch |
| `validation/epoch_NNN.npz` | validation 전체 sample ID/label/10-class logits | 매 epoch |
| `probe/epoch_NNN.npz` | 고정200장 GT coverage/FG, raw attention196, raw/actual 선택 indices, 실제 swaps, student logits, teacher full/raw/actual logits, full teacher attention | 0 및 매 epoch |
| `probe/epoch_NNN_counterfactual.npz` | 동일 student에서 6가지 teacher 입력 방식을 5회 반복한 indices/swaps/logits | 0/10/25/50/75/100 |
| `probe/best*.npz` | best checkpoint의 같은 진단 | 완료 |
| `epoch_NNN.pt` | 가중치와 provenance; optimizer 없음 | 0/10단위/final |
| `best.pt` | val macro로 선택한 weights/provenance | best 갱신 |
| `last.pt` | weights + optimizer + AMP scaler + best weights + history | 매 epoch 교체 |
| `resume_050.pt` | 후반 분기용 full 상태 | MaskedKD50에만 |
| `test_best.npz`, `test_last.npz`, `test_metrics.json` | 최종 test 이미지별 예측/정답 및 macro/class별 결과 | 명시적 evaluate |

`probe`의 random sample은 같은 image/method/repeat이면 seed와 epoch에 무관하게 같게 만든다. 이것은 훈련 random swap을 고정한다는 뜻이 아니다. **훈련에서는 각 step에서 새로 샘플링**한다. Probe의 `actual`은 해당 학습 방법을 고정 검증 이미지에 적용한 것으로, 훈련 시 augmentation된 이미지의 모든 mask를 기록한 것은 아니다.

생성 표·그림:

- `analysis/learning_curves.csv`: validation·학습 시간과 LR.
- `analysis/probe_curves.csv`: 교체 전/후 선택·teacher 손상·student 모방의 epoch별 평균.
- `analysis/counterfactual.csv`: 같은 checkpoint에서 마스크만 바꾼 5회 반복 비교.
- `analysis/error_correction.csv`: 동일 시작 오류 집합의 교정 시점/비율. 미교정 샘플을 삭제하지 않는다.
- `analysis/convergence.csv`: validation macro 80%를 3회 연속 달성한 첫 epoch·update·시간, 미도달 여부, 최근20 epoch 비교.
- `analysis/test_results.csv`, `paired_seed_differences.csv`: best/last 분리 결과 및 seed-paired 효과 크기.
- `analysis/per_image/*.csv.gz`: 매 epoch/image의 선택률, logits 기반 KL, 예측 label/정오, 교정기회, swaps. GitHub에서 후속 통계 분석 가능.
- `analysis/per_image_counterfactual/*.csv.gz`: 같은 checkpoint에서 mask별·반복별·이미지별 teacher 변화. Paired mask 비교용이며 반복을 독립 seed로 세지 않는다.
- `analysis/main_scratch.png/svg`, `main_imagenet.png/svg`: val 성능, teacher KL, raw BSR/FMR 곡선(평균±seed SD).

`export --push`에는 위 분석 결과, config/history/test 원예측, dataset manifest, storage/환경 기록, checkpoint 경로/크기/hash 목록을 포함한다. `*.pt`, 전체 attention/probe NPZ, 이미지 원본은 서버에 둔다. 따라서 **보고서 push는 checkpoint 백업이 아니다.** 서버 손실에 대비한 가중치 백업은 별도 디스크/스토리지로 수행해야 한다.

무엇까지 알 수 있는가:

- 저장된 모든 epoch의 선택 변화·교정·teacher 손상·validation 성능은 비교 가능.
- 가중치를 남긴 epoch에서는 새로운 이미지·마스크로 재평가 가능.
- 임의 중간 epoch의 새로운 개입·gradient 재분석은 그 epoch 가중치가 없으면 불가능.
- 가중치만 저장한 epoch에서 optimizer 상태까지 동일한 재학습 분기를 복원할 수는 없다. full 상태가 있는 last/MaskedKD50을 사용한다.
- 현재 분석은 배경을 직접 교체한 causal robustness 평가, 모든 head/layer attention, step별 gradient, teacher size ablation을 수행하지 않는다. 필요한 경우 저장 weights와 원본 images로 별도 실험한다.

**Teacher prediction quality와 student supervision quality를 구분한다.** FG oracle은 GT를 강제로 사용하고, 개선된 teacher 정확도 자체가 좋은 증류를 보장하지 않는다. 주 결과는 student macro accuracy와 오류 교정이며, 선택/teacher 지표는 원인을 이해하기 위한 보조 지표다.
