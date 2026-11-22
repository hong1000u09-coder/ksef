# 27개 보정 후보 선호 실험

기존 프로젝트와 분리된 독립 구현입니다. 기존 `Untitled-1.py`는 수정하지
않았습니다. 사전 학습된 취향 가중치나 생성한 응답 라벨은 포함하지 않습니다.

## 설치 및 실행

Python 3.10 이상을 사용합니다.

```powershell
py -m pip install -r requirements.txt
py personalization.py run --data data.json --config config.json --out results
```

`data.json`에 예시 응답은 넣지 않았습니다. 실제 참가자 응답만 입력하세요.
`tests/test_personalization.py`의 합성 데이터는 기능 테스트에만 사용합니다.

실험을 각각 실행할 수 있습니다.

```powershell
py personalization.py run --data data.json --config config.json --out results --experiment A
py personalization.py run --data data.json --config config.json --out results --experiment B
```

실험 A는 평가 사용자별로 동일한 응답을 P1과 P2에 제공해 P0/P1/P2를 비교합니다.
P0는 독립적인 `developer` 사용자의 질문용 데이터 응답으로만 학습합니다. 그런
데이터가 없으면 결과를 만들지 않습니다. 실험 B는 P1을 고정하고 Random,
Balanced Random, Predictive Entropy, BALD를 비교합니다. P2에서도 별도 비교를
수행하려면 설정 파일에서 `"strategy_models": ["P1", "P2"]`로 지정합니다.

원본 그룹 분할은 `split_seed`와 `evaluation_fraction`으로 재현할 수 있습니다.
평가용 `leakage_group_id`를 `evaluation_group_ids`로 직접 지정할 수도 있습니다.
알려진 근접 중복 원본은 같은 `leakage_group_id`를 지정해야 합니다. 그룹 ID가
없으면 원본별로 따로 취급하며, 프로그램이 이미지 중복을 자동 탐지하지는 않습니다.

## 입력 데이터 형식

최상위 JSON 객체에 `candidates`, `originals`, `questions`, `users`, `responses`
배열을 둡니다.

- 후보: `{"id":"c00","adjustment":[색온도,대비,채도]}` 형식입니다. 서로 다른
  27개 보정값을 제공해야 합니다. 실제 렌더러가 픽셀 동일 결과를 확인할 수 있으면
  모든 후보에 `render_signature`도 제공하세요. 중복 서명은 거부됩니다. 서명을
  제공하지 않으면 보정값이 완전히 동일한 후보만 중복 검사할 수 있습니다.
- 원본: `{"id":"img01","leakage_group_id":"group01","scene_label":"실내",
  "source_features":{"brightness_mean":0.41,"saturation_mean":0.28}}` 형식입니다.
  장면 라벨과 수치 특징은 원본 이미지나 기록된 메타데이터에서 가져와야 합니다.
  보정 후보 렌더링의 특징을 원본 특징으로 복사하면 안 됩니다. Balanced Random에
  쓰는 채도 평균은 `[0,1]` 범위여야 합니다.
- 질문: `{"id":"q01","original_id":"img01","candidate_a_id":"c00",
  "candidate_b_id":"c01"}` 형식입니다. 같은 원본에서 실제 비교할 A/B 후보를
  정의합니다. 같은 원본의 순서만 뒤집은 중복 후보쌍은 거부됩니다.
- 사용자: `{"id":"dev01","role":"developer"}` 또는
  `{"id":"eval01","role":"evaluation"}` 형식입니다. 역할은 응답 수로 추정하지 말고
  연구 프로토콜에 따라 지정해야 합니다.
- 응답: `{"user_id":"eval01","question_id":"q01",
  "chosen_candidate_id":"c01","response_ms":1730}` 형식입니다. 선택은 화면의
  좌우가 아니라 후보 ID로 기록합니다.

P1/P2는 질문용 분할 원본에 대한 평가 사용자의 실제 응답만 학습에 사용합니다.
평가 분할 응답은 점수 계산에만 씁니다. P0는 질문용 분할에 속한 개발 사용자의
응답만 학습합니다. 알려진 동일 누출 그룹의 원본은 같은 분할에 둡니다. 분할 그룹이나
실제 응답이 부족하면 결과 보고서에 평가 불가 범위를 표시하며 라벨을 만들거나
시드 수를 참가자 수처럼 취급하지 않습니다.

## 모델과 질문 선택법

- **P0**: 개발 사용자 응답으로만 학습하는 사용자 공통 9개 특징 베이지안 로지스틱
  모델입니다.
- **P1**: 사용자별 9개 특징 베이지안 로지스틱 모델입니다. 사진 맥락 특징을 쓰지
  않으므로 원본 사진이 달라도 후보 순위가 같을 수 있습니다.
- **P2**: P1 특징에 `보정값 × z(원본)` 상호작용을 추가합니다. `z`는 설정한 원본
  수치 특징의 표준화 값과 선택적 장면 원-핫 특징으로 구성됩니다. 수치 표준화와
  장면 범주 선택은 질문용 원본만 사용합니다. P1/P2는 가우시안 사전분포와 Laplace
  사후 근사를 함께 사용합니다. `interaction_prior_std`를 작게 설정해 P2 상호작용을
  강하게 정규화하지만, 사용자별 계수가 늘어 과적합할 수 있습니다.
- **Random**: 공통 질문 풀의 남은 질문에서 균등 무작위 선택합니다.
- **Balanced Random**: 남은 `scene_label × saturation_mean` 조건별 질문 수를
  비교해 가장 적은 조건을 고른 뒤 그 조건 안에서 질문을 균등 선택합니다. 채도 구간은
  설정 파일의 경계값을 사용하고, 가장 적은 조건이 여럿이면 조건을 균등 무작위로
  고릅니다.
- **Predictive Entropy**: Laplace 사후 표본에서 추정한 사후예측 평균 확률의 엔트로피를
  최대화합니다.
- **BALD**: 사후예측 평균의 엔트로피에서 사후 표본별 엔트로피 평균을 뺀 값을
  최대화합니다.

모든 선택기는 실제 응답이 있는 동일한 남은 질문 풀에서
`(original_id, candidate_A, candidate_B)` 질문 하나를 선택합니다. 그 풀 밖에서
새 후보쌍을 만들거나 특정 선택기만 더 넓은 풀을 사용하는 것은 아닙니다. 실험 B에서는
기록된 장면 라벨 또는 원본 채도값이 없거나 채도값이 `[0,1]` 범위가 아닌 질문을
모든 선택기에 공통으로 제외합니다. 질문을 고른 다음에야 해당 응답을 공개합니다.
오프라인 재생은 실제 응답이 기록된 질문에 한정되며, 묻지 않은 질문의 응답을
인과적으로 추정하지 않습니다.

## 결과 파일

실행 결과 폴더에 다음 파일을 기록합니다.

- `split_manifest.csv`: 원본별 누출 그룹 및 분할 기록
- `question_order.csv`, `training_log.csv`: 질문 순서, 표시 후보 연결, 응답·선택
  시간, 실제 학습 응답 ID
- `predictions.csv`, `metrics.csv`: 평가 예측 확률, 로그손실, 정확도, Brier 점수와
  사용자별 및 사용자 평균 결과
- `budget_curve.png`: 응답 예산별 모델/선택기 곡선. 시드는 반복 실행이며 독립
  참가자로 계산하지 않습니다.
- `recommendations.csv`: 27개 후보 중 사후평균 점수가 가장 높은 후보
- `blind_recommendation_queue.csv`: 실제 블라인드 A/B 추천 비교를 위한 후보쌍.
  동일 추천도 별도 표시하며, 모델 점수를 만족도 정답으로 취급하지 않습니다.
- `report.md`: 데이터 충족 여부, 실행 결과 및 해석 제한

## 온라인 수집 및 모델 업데이트

참가자를 전략에 무작위 배정합니다. 같은 참가자에게 다시 배정 명령을 실행해도
기존 배정을 유지합니다.

```powershell
py personalization.py assign --participant-id p001 --arms Random "Balanced Random" "Predictive Entropy" BALD --seed 13
```

한 참가자가 여러 전략을 사용하도록 사전에 계획한 교차 실험은 순서를 무작위화해
기록합니다. 응답 이력은 사용자·전략·세션별로 분리됩니다.

```powershell
py personalization.py assign-crossover --participant-id p002 --arms Random BALD --seed 13
```

실제 화면에 표시된 좌우 후보 ID로 응답을 기록합니다. `next` 명령은 무작위 배정과
동일한 사용자·전략·세션의 이력을 확인해 질문을 선택합니다. 화면에서 A/B 순서가
바뀌더라도 선택은 후보 ID에 연결됩니다.

```powershell
py personalization.py next --data data.json --config config.json --user-id p001 --strategy-id Random --session-id s001 --model P1 --seed 13
py personalization.py record --data data.json --user-id p001 --strategy-id Random --session-id s001 --original-id img01 --question-id q01 --left-id c01 --right-id c00 --selected-side left --response-ms 1700
```

블라인드 추천 비교는 `--question-id`를 생략하고 큐의 후보 ID를 사용해
`--comparison-kind recommendation --comparison-id <queue-id>`로 기록합니다.
두 후보가 같으면 `--selected-side tie`로 기록해야 합니다. 실제 응답 결과를 분석합니다.

```powershell
py personalization.py analyze-recommendations --events online/responses.jsonl --queue results/blind_recommendation_queue.csv --out results/recommendation_outcomes.csv
```

사용자별 모델 업데이트는 같은 사용자·전략·세션의 실제 선호 응답만 사용합니다.

```powershell
py personalization.py update --data data.json --config config.json --user-id p001 --strategy-id Random --session-id s001 --model P1 --out online/p001-random-p1.json
```

추천 비교 응답은 개인화 학습에 포함되지 않습니다. 전략을 바꾸거나 여러 전략에
참여할 때는 세션과 순서 효과를 따로 관리해야 합니다.
