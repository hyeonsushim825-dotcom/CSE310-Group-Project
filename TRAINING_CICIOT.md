# CICIoT2023: 채연 담당 학습 코드

기준: 팀 저장소 main `0879aff8540ce659c4936c77ef89fafcbe258f86`.
이 묶음은 **학습 파이프라인**이며, 평가/필터링 전체 구현은 현수 담당으로 남긴다.
실제 Drive 데이터 및 서버 GPU에서의 학습은 아직 실행하지 않았다.
서버 주소/계정/비밀번호는 코드에 넣지 않는다.

## 1. 지금 GitHub에 올릴 여섯 파일

ZIP을 풀고 다음 파일들을 저장소 **최상위**에 넣는다. data_preprocessing/ 안에 넣지 않는다.

| 파일 | 처리 | 역할 |
|---|---|---|
| train_ciciot.py | 추가 | 학습, 체크포인트, 중단/재개, 중심/점수 계산 |
| check_ciciot_data.py | 추가 | 실제 배열 shape/유효값/선택적 라벨 정렬 검사 |
| test_ciciot_training.py | 추가 | GPU/실제 데이터 없이 실행하는 CPU 테스트 |
| requirements-ciciot.txt | 추가 | 새 코드의 최소 의존성 |
| TRAINING_CICIOT.md | 추가 | 이 안내와 팀원 간 인터페이스 |
| .gitignore | 업데이트 | 기존 내용 보존 + datasets/runs/NPY/NPZ 제외 |

앞서 받은 train_ciciot.py/TRAINING_CICIOT.md가 있다면 이번 버전으로 대체한다.
기존 train_clan.py, eval_clan.py, finetune_clan.py, requirements.txt 및
data/, model/, loss/, util/, data_preprocessing/는 삭제하지 않는다.
새 데이터 학습은 `python train_ciciot.py ...`로 실행한다. 기존 train_clan.py는 Lycos용이다.

.gitignore는 위 commit 기준의 원본에 아래 항목만 추가했다. 다른 팀원이 그 후 수정했다면
통째로 덮지 말고 현재 파일 끝에 아래 네 줄만 추가한다.

```gitignore
/datasets/
/runs/
*.npy
*.npz
```

## 2. GitHub 웹 업로드

1. 팀 repo 페이지를 연다.
2. main 브랜치 선택 메뉴에서 `chaeyeon/ciciot-training` 브랜치를 만든다. 이미 있으면 선택한다.
3. 최상위에서 Add file → Upload files를 선택한다.
4. ZIP 자체/상위 폴더가 아니라 압축을 푼 **여섯 파일**을 올린다.
5. commit 메시지: `Add CICIoT2023 training pipeline and validation tests`
6. 해당 브랜치에 commit한다.
7. Compare & pull request로 main에 합칠 PR을 만든다. 현수와 확인 후 merge한다.

이 준비 작업에서는 원격 저장소에 push/PR 생성/merge하지 않았다.
main에 merge 전이라면 서버에서 main만 받았을 때는 새 파일이 보이지 않는다.

## 3. 로컬 Git으로 올리기 (웹 방법과 둘 중 하나)

팀 repo 폴더에서 `git status`로 미완료 작업을 먼저 확인한다.
작업 폴더가 깨끗하다면:

```bash
git switch main
git pull --ff-only origin main
git switch -c chaeyeon/ciciot-training
```

여섯 파일을 최상위에 복사한 뒤:

```bash
git add train_ciciot.py check_ciciot_data.py test_ciciot_training.py requirements-ciciot.txt TRAINING_CICIOT.md .gitignore
git diff --cached --stat
git commit -m "Add CICIoT2023 training pipeline and validation tests"
git push -u origin chaeyeon/ciciot-training
```

브랜치가 이미 있으면 `git switch chaeyeon/ciciot-training`으로 이동한다.
미완료 작업을 삭제/reset하지 않는다. 데이터, 실행 결과, 비밀번호는 commit하지 않는다.

## 4. Windows PC에서 지금 가능한 테스트

Python 3.10 이상이 필요하다. 실제 데이터와 GPU는 필요 없다.
PowerShell을 repo 폴더에서 열고:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ciciot.txt
.\.venv\Scripts\python.exe -m unittest -v test_ciciot_training
```

CPU 전용 PyTorch를 원하면 https://pytorch.org/get-started/locally/ 에서 Windows/Pip/Python/CPU
명령을 골라 같은 환경에 먼저 설치한다. GPU 서버의 설치는 서버 환경 확인 후 따로 진행한다.
기존 requirements.txt로 다시 설치하면 버전이 바뀔 수 있으므로 새 학습에는 별도 목록을 쓴다.
테스트는 임시 폴더에 합성 데이터/모델을 만들고 종료 시 지운다. 성공하면 `OK`가 나온다.

## 5. 실제 데이터 검사 (파일을 받은 후)

repo 최상위 `datasets/`에 다음 파일을 둔다.

| 파일 | shape |
|---|---|
| benign_train_200k_scaled.npy | (200000,39) |
| train_5pct_contaminated.npy | (210526,39) |
| train_10pct_contaminated.npy | (222222,39) |

```bash
python check_ciciot_data.py --data-dir datasets --report runs/data_check.json
```

같은 폴더에 train_5pct_labels.npy/train_10pct_labels.npy도 있다면 선택적으로:

```bash
python check_ciciot_data.py --data-dir datasets --check-labels --report runs/data_check_with_labels.json
```

라벨 검사는 0/1 여부, 공격 수(10526/22222), 정상으로 표시된 행이 고정 정상 20만 행과
같은지 확인한다. 데이터 QA에만 라벨을 쓰며 학습은 라벨을 읽지 않는다. 입력 파일은 바꾸지 않는다.
shape 검사를 통과해도 feature 이름/순서, scaler fit 대상, train/test 누수까지 확인된 것은 아니다.
전처리 담당자에게 feature 목록/순서, scaler와 정상 split/20만 추출 코드 및 seed를 함께 받는다.

## 6. 나중에 사용할 학습 명령 (지금 실행할 필요 없음)

GPU 환경이 확인되면 작은 테스트부터:

```bash
python -u train_ciciot.py --train-npy datasets/benign_train_200k_scaled.npy --run-dir runs/smoke_clean --max-samples 4096 --epochs 2 --batch-size 256 --device cuda
```

본 실험 시작 설정은 200 epoch, batch 1024, seed 42이며 최적값을 보장하지 않는다.
전체 데이터 2 epoch의 메모리/시간을 먼저 확인하고 비교군 설정을 통일한다.
본 실험에서는 --max-samples를 빼야 한다. 매 실행 새 모델을 학습한다.

```bash
python -u train_ciciot.py --train-npy datasets/benign_train_200k_scaled.npy --run-dir runs/clean_s42 --epochs 200 --batch-size 1024 --seed 42
python -u train_ciciot.py --train-npy datasets/train_5pct_contaminated.npy --run-dir runs/c05_s42 --epochs 200 --batch-size 1024 --seed 42
python -u train_ciciot.py --train-npy datasets/train_10pct_contaminated.npy --run-dir runs/c10_s42 --epochs 200 --batch-size 1024 --seed 42
```

첫 실행 성공을 확인한 뒤 다음 것을 실행한다. 공유 GPU 사용 시간/방식은 관리자와 맞춘다.
추가 seed는 43,44 등을 사용하고 run-dir도 s43,s44로 구별한다.
출력: config.json, loss.csv, last.pt, final.pt, centroid.npy.
final.pt는 학습 및 최종 중심 계산이 끝났을 때 생긴다. last.pt는 마지막 완료 epoch의 상태다.
config/checkpoint에는 실제 torch/numpy 버전, 데이터 SHA256, 관련 코드 SHA256, 설정을 남긴다.
별도로 `python -m pip freeze` 출력도 실험 결과와 함께 보관한다.

## 7. 중단 후 이어 학습

처음과 같은 명령에 --resume만 붙인다. 예:

```bash
python -u train_ciciot.py --train-npy datasets/train_5pct_contaminated.npy --run-dir runs/c05_s42 --epochs 200 --batch-size 1024 --seed 42 --resume
```

마지막 완료 epoch 다음부터 재개한다. 중간에 끊긴 epoch는 다시 수행한다.
데이터/학습 설정/관련 코드/torch 및 numpy 버전이 바뀌면 재개를 거부한다.
--epochs는 남은 횟수가 아니라 원래 전체 목표다. LR schedule 유지를 위해 바꾸지 않는다.
final.pt가 있으면 완료된 실험이므로 재개하지 않는다.
계획적으로 멈추려면 처음에 `--stop-after-epoch 10` 등을 넣고, 재개 때 빼고 --resume를 넣는다.
새 비교 실험이나 필터링 후 재학습에는 --resume를 사용하지 않는다.

## 8. 현수와 공유할 인터페이스

```python
from train_ciciot import load_features, load_trained_model, anomaly_scores

model, ckpt = load_trained_model('runs/c05_s42/final.pt', device='cuda')
x_train = load_features('datasets/train_5pct_contaminated.npy')
scores = anomaly_scores(model, x_train, ckpt['centroid'], 'cuda', batch_size=1024)
```

- `model(x)`는 원래 ContrastiveMLP의 **64차원 projection**이다. forward_features의 1024차원과 혼용하지 않는다.
- centroid는 해당 모델의 전체 학습 행에서 얻은 raw projected embedding의 평균이다.
- score = 1 - cosine_similarity이며 클수록 의심스럽다. 원래 -cosine과 순위는 같다.
- scores[i]는 입력의 i번째 행과 대응한다. 추론 시 순서를 섞지 않는다.
- 실제 라벨로 정상만 골라 중심을 만들지 않는다. 각 모델은 자기 학습 데이터의 중심을 쓴다.
- test도 같은 scaler/feature 순서로 준비하여 anomaly_scores에 넣는다.
- test label과 점수로 AUROC/AUPRC/유형별 결과를 계산하는 부분은 현수가 구현한다.
- 기존 eval_clan.py의 CSV/72차원/정상-label 기반 중심 코드는 그대로 사용하면 안 된다.
- n_classes=None: self-supervised 학습에는 classifier head가 필요 없다.

필터링은 제거 index를 저장하고 정답 label은 선정 이후 제거 precision/recall 분석에만 쓴다.
random/distance 제거는 동일 원본에서 동일한 개수 k를 제거한다. 출력은 scaled (N,39) NPY다.
다시 scaling하지 않는다. 다음 파일명은 제안하는 출력 계약이며 현재 파일이 있다는 뜻은 아니다.

| 필터링 출력 | 재학습 run-dir |
|---|---|
| datasets/train_5pct_distance_s42.npy | runs/c05_distance_s42 |
| datasets/train_5pct_random_s42.npy | runs/c05_random_s42 |
| datasets/train_10pct_distance_s42.npy | runs/c10_distance_s42 |
| datasets/train_10pct_random_s42.npy | runs/c10_random_s42 |

```bash
python -u train_ciciot.py --train-npy datasets/train_5pct_distance_s42.npy --run-dir runs/c05_distance_s42 --epochs 200 --batch-size 1024 --seed 42
```

입력 경로만 바꿔 같은 학습 파일을 사용한다. 7조건 x 3 seed면 21회 학습이다.
distance filtering은 seed별 초기 모델로 다시 계산하고 random filtering seed도 기록한다.

## 9. 실험 해석 및 검증 범위

- 최신 데이터는 정상 20만 고정 + 공격 추가다. 전체 크기 고정이라는 옛 문구를 수정한다.
- 같은 epoch라도 총수가 달라 update 수가 다르며 config에 기록된다.
- drop_last=True: 매 epoch 마지막 소수 행은 생략하고 다음 epoch에 다시 섞는다. 중심은 모든 행으로 계산한다.
- 모델/loss/uniform augmentation/LR schedule을 원본에서 재사용한다. 원 논문 수치의 완전 재현을 뜻하지 않는다.
- 입력은 정상 train에서 fit한 scaler로 이미 변환됐다고 가정한다. 여기서는 재분할/정규화/label filtering하지 않는다.
- 5%오염에서 5%제거, 10%오염에서 10%제거는 알려진 오염 budget 실험이라고 명시한다.
- test/validation은 고정한다. attack-type label을 보존하고 pooled AUROC와 유형별 AUROC 평균을 구분한다.
- AUPRC는 평가 클래스 비율을 기록한다. FPR threshold는 validation에서 정하고 test에서는 고정한다.
- 거리 filtering 비교는 첫 단계이며 CLAN-specific 개선법까지 완성된 것은 아니다.
- 실제 Drive 파일, TITAN RTX CUDA/메모리/시간, 실제 탐지 성능은 아직 별도 검증이 필요하다.
- CPU 합성 데이터 테스트 6개: 기본 모델 forward/backward, 입력/라벨 QA, checkpoint/모든 행 중심/점수 행 순서,
  동일 seed 재현성 및 중단/재개 일치, 덮어쓰기/잘못된 resume 거부 등을 확인한다.
  다른 GPU/라이브러리 환경 간 비트 단위 동일성은 보장하지 않는다.
