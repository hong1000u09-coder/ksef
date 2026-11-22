# 27-candidate preference experiments

This is a new, standalone implementation. The existing `Untitled-1.py` is left
untouched. No pretrained preference weights or generated response labels are
included.

## Install and run

Use Python 3.10+:

```powershell
py -m pip install -r requirements.txt
py personalization.py run --data data.json --config config.json --out results
```

`data.json` is intentionally not populated with example answers. Only real,
recorded participant responses belong in it. The synthetic fixture in
`tests/test_personalization.py` is used exclusively for feature tests.

Run one experiment:

```powershell
py personalization.py run --data data.json --config config.json --out results --experiment A
py personalization.py run --data data.json --config config.json --out results --experiment B
```

Experiment A compares P0, P1, and P2 after giving P1/P2 the same response rows
per evaluation user. P0 is fit only from independent `developer` users'
question-partition responses. It is not reported if those data do not exist.
Experiment B compares Random, Balanced Random, Predictive Entropy, and BALD
with P1 fixed; set `"strategy_models": ["P1", "P2"]` to run a separate P2
selector comparison as well.

The deterministic group split uses `split_seed` and `evaluation_fraction`, or
you can fix the held-out `leakage_group_id` values in `evaluation_group_ids`.
Known near-duplicate originals must be assigned the same
`leakage_group_id`. Missing group IDs default to one group per original; the
program does not detect near-duplicate images.

## Input data contract

The root JSON object has `candidates`, `originals`, `questions`, `users`, and
`responses` arrays:

- Candidate: `{"id":"c00","adjustment":[temperature,contrast,saturation]}`.
  Supply exactly 27 unique adjustments. Add `render_signature` when the real
  renderer can identify pixel-identical outputs; duplicate signatures are
  rejected. Without signatures, only identical adjustment vectors can be
  checked for duplicate rendering.
- Original: `{"id":"img01","leakage_group_id":"group01","scene_label":"indoor",
  "source_features":{"brightness_mean":0.41,"saturation_mean":0.28}}`.
  Scene labels and numeric features must come from the original image/recorded
  metadata. Do not copy features from an adjusted candidate render. The
  configured saturation mean is expected in `[0,1]` for Balanced Random.
- Question: `{"id":"q01","original_id":"img01","candidate_a_id":"c00",
  "candidate_b_id":"c01"}`. This defines the actual shown A/B pair on one
  original. Reversed duplicate pairs for the same original are rejected.
- User: `{"id":"dev01","role":"developer"}` or
  `{"id":"eval01","role":"evaluation"}`. Roles must be assigned by the study
  protocol, not inferred from response counts.
- Response: `{"user_id":"eval01","question_id":"q01",
  "chosen_candidate_id":"c01","response_ms":1730}`. Choice is tied to candidate
  identity, never to the screen side.

P1/P2 user updates may use only evaluation-user responses whose originals are
in the question partition. Evaluation-partition responses are scoring labels
only. P0 can use only developer-user responses in the question partition.
An original and all rows sharing its known leakage group stay in one partition.
If there are too few groups or real responses, the report names the unavailable
claims rather than manufacturing labels or treating seeds as participants.

## Models and selection

- P0: one shared nine-feature Bayesian logistic model, updated with developer
  responses only.
- P1: per-user nine-feature Bayesian logistic model. It has no image context
  and can rank candidates identically across different originals.
- P2: P1 features plus `adjustment × z(original)`. `z` contains configured
  standardized original numeric features and optional scene one-hot features.
  Numeric standardization and scene vocabulary use question-partition originals
  only. Gaussian priors and a Laplace posterior approximation are used by P1
  and P2; the smaller configured `interaction_prior_std` strongly regularizes
  P2. Its per-user context coefficients still add overfitting risk.
- Random samples uniformly from the common remaining answered-question pool.
- Balanced Random chooses uniformly among the least-populated remaining
  `scene_label × saturation_mean` buckets, then uniformly within that bucket.
  Bucket edges are in config; tied buckets are selected uniformly.
- Predictive Entropy maximizes entropy at the posterior-predictive mean
  probability, estimated from Laplace posterior samples.
- BALD maximizes entropy of the posterior-mean prediction minus mean
  posterior-sample entropy.

Each selector chooses a whole question from the same finite pool of real
answered questions. It does not generate candidate pairs or get a broader
candidate pool. For Experiment B, questions without recorded scene and
in-range original saturation are excluded once from the shared pool for every
selector. An answer is revealed only after the selector returns its question.
Offline replay is conditional on questions with actual responses and is not a
causal estimate for unasked questions.

## Outputs

Each run writes:

- `split_manifest.csv`: original-to-leakage-group partition record.
- `question_order.csv`, `training_log.csv`: selected questions, displayed
  candidate mapping, response/selection times, and actual training IDs.
- `predictions.csv`, `metrics.csv`: held-out probabilities and log loss,
  accuracy, Brier score, with per-user and participant-averaged rows.
- `budget_curve.png`: model and strategy curves; seeds are repeated runs, not
  independent participants.
- `recommendations.csv`: posterior-mean-score top candidate among all 27.
- `blind_recommendation_queue.csv`: candidate pairs for collecting real,
  blinded recommendation comparisons. Identical recommendations are retained
  and flagged; model utilities are never used as satisfaction labels.
- `report.md`: data sufficiency, completion, and interpretation limits.

## Online collection and updates

Assignment is randomized per participant and idempotent for a participant:

```powershell
py personalization.py assign --participant-id p001 --arms Random "Balanced Random" "Predictive Entropy" BALD --seed 13
```

For an explicitly planned crossover, randomize and persist sequence order.
The response history remains isolated by user, strategy, and session:

```powershell
py personalization.py assign-crossover --participant-id p002 --arms Random BALD --seed 13
```

Record the actual displayed left/right candidate identities. The assignment
log is checked and the response maps back to candidate ID even if the UI swaps
the question's A/B order:

```powershell
py personalization.py next --data data.json --config config.json --user-id p001 --strategy-id Random --session-id s001 --model P1 --seed 13
py personalization.py record --data data.json --user-id p001 --strategy-id Random --session-id s001 --original-id img01 --question-id q01 --left-id c01 --right-id c00 --selected-side left --response-ms 1700
```

To collect a blind recommendation comparison, omit `--question-id`, provide
`--comparison-kind recommendation --comparison-id <queue-id>`, and use the
candidate IDs from the queue. Same-candidate comparisons must be recorded as
`--selected-side tie`. Analyze actual recorded comparisons (not model scores):

```powershell
py personalization.py analyze-recommendations --events online/responses.jsonl --queue results/blind_recommendation_queue.csv --out results/recommendation_outcomes.csv
```

Update a user posterior from only that user/strategy/session's logged
preference events:

```powershell
py personalization.py update --data data.json --config config.json --user-id p001 --strategy-id Random --session-id s001 --model P1 --out online/p001-random-p1.json
```

Recommendation-comparison events are never used for personalization updates.
Different strategies and sessions have isolated histories. Keep session IDs
and order effects separate when a participant is deliberately exposed to
multiple strategies.
