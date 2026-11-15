# CLAN Filtering 사용 안내

## 1. 코드 역할

`filter_training.py`는 학습 데이터에서 일부 샘플을 제거하고, 재학습용 데이터를 저장합니다.

| 방법 | 제거 기준 |
|---|---|
| Distance filtering | 학습된 모델의 임베딩 중심에서 cosine distance가 큰 샘플부터 제거 |
| Random filtering | 같은 원본 데이터에서 동일한 개수를 무작위로 제거 |

두 방법 모두 정상·공격 label을 사용하지 않습니다.

## 2. 실행 준비

- `filter_training.py`를 `train_ciciot.py`와 같은 폴더에 넣습니다.
- Distance filtering에는 해당 데이터로 학습한 `final.pt`가 필요합니다.
- 모델 설정과 중심은 자동으로 불러오므로 학습 코드를 수정할 필요는 없습니다.

## 3. 실행 방법

### 5% 오염 데이터

```bash
python filter_training.py --data datasets/train_5pct_contaminated.npy --checkpoint runs/c05_s42/final.pt --remove-ratio 0.05 --method both --prefix train_5pct --seed 42 --output runs/filter_c05_s42 --device cuda
```

### 10% 오염 데이터

```bash
python filter_training.py --data datasets/train_10pct_contaminated.npy --checkpoint runs/c10_s42/final.pt --remove-ratio 0.10 --method both --prefix train_10pct --seed 42 --output runs/filter_c10_s42 --device cuda
```

### 주요 옵션

| 옵션 | 의미 |
|---|---|
| `--method both` | Distance와 random 모두 실행 |
| `--method distance` | Distance만 실행 |
| `--method random` | Random만 실행. `--checkpoint` 생략 가능 |
| `--remove-ratio` | 제거할 비율 |
| `--seed` | Random 제거의 재현성을 위한 값 |
| `--device cpu` | GPU 없이 실행 |

이번 실험은 5% 오염 데이터에서 5%, 10% 오염 데이터에서 10%를 제거하는 기준으로 진행합니다. 팀 논의로 변경되면 --remove-ratio를 수정하세요.

## 4. 결과 파일

5% 예시는 `runs/filter_c05_s42/`에 다음 파일을 저장합니다.

| 파일 | 용도 |
|---|---|
| `train_5pct_distance_s42.npy` | Distance filtering 후 재학습용 데이터 |
| `train_5pct_random_s42.npy` | Random filtering 후 재학습용 데이터 |
| `*_kept_indices.npy` | 남긴 행 번호 |
| `*_removed_indices.npy` | 제거한 행 번호 |
| `distance_scores.npy` | 원본 행 순서에 대응하는 거리 점수 |
| `center.npy` | Filtering에 사용한 중심 |
| `summary.json` | 실행 설정, 제거 개수 등 |

저장된 데이터는 이미 scaling된 상태이므로 다시 scaling하지 않습니다.

## 5. 재학습

각 결과 파일을 사용해 새 모델을 학습합니다.
10% 데이터의 재학습은 아래 재학습 명령어에서 다음 경로를 바꿔 실행합니다.
- `filter_c05_s42` → `filter_c10_s42`
- `train_5pct` → `train_10pct`
- `c05_distance_s42` → `c10_distance_s42`
- `c05_random_s42` → `c10_random_s42`

### Distance 결과로 재학습

```bash
python -u train_ciciot.py --train-npy runs/filter_c05_s42/train_5pct_distance_s42.npy --run-dir runs/c05_distance_s42 --epochs 200 --batch-size 1024 --seed 42
```

### Random 결과로 재학습

```bash
python -u train_ciciot.py --train-npy runs/filter_c05_s42/train_5pct_random_s42.npy --run-dir runs/c05_random_s42 --epochs 200 --batch-size 1024 --seed 42
```

두 방법의 학습 조건을 동일하게 맞추고, `--resume`는 사용하지 않습니다.

## 6. 주의사항

- 두 방법은 같은 원본 데이터에서 독립적으로 같은 개수를 제거합니다.
- 제거 비율은 미리 정하거나 validation으로 결정합니다. Test 결과로 선택하지 않습니다.
- 오염 비율과 동일한 제거 비율을 사용하면, 오염 비율을 알고 설정한 실험임을 보고서에 명시합니다.
- Label은 제거가 끝난 뒤 정상·공격 제거 결과를 분석할 때만 사용합니다.
- 재학습 모델 평가에는 재학습 후 `final.pt`의 새 중심을 사용합니다.
- 기존 출력 폴더는 덮어쓰지 않습니다. 다시 실행할 때는 새 폴더를 지정합니다.
- 다른 seed로 실험할 때는 해당 seed로 학습한 모델을 사용하고, filtering의 seed와 출력 경로도 바꿉니다.

## 7. 테스트 상태

작은 CPU 테스트에서 학습 → filtering → 재학습과 데이터·index 대응을 확인했습니다.

실제 전체 데이터와 GPU 실행, 탐지 성능 평가는 아직 진행하지 않았습니다.