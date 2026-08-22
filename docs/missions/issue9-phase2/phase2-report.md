# Issue #9 Phase 2 Report

## Scope and Reproducibility

The final Phase 2 run used the fixed command supplied by the Task 6 report
instructions on a GPU machine at `~/beme-p2` on 2026-08-22 (JST). No
re-measurement was performed for this report.

Recorded commit: `19211945e514efc03fc832d6071e5689b5aa3dfe` (current checkout
HEAD; the supplied run was described as `9b0510b` or later).

The exact CLI arguments were:

```bash
uv run python scripts/train_scorer.py \
  --train data/d1/train.jsonl --dev data/d1/dev.jsonl --test data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --initial-weights data/weights/decomposed_v1.json \
  --epochs 5 --learning-rate 0.1 --seed 9 \
  --output-weights artifacts/issue9-phase2/run1/weights.json \
  --output-report artifacts/issue9-phase2/run1/report.json
```

The same command was run for `run2`. Both `weights.json` and `report.json`
were byte-identical between the two runs (`cmp` matched).

## Data and Provenance

| Split | Examples | Provenance |
|---|---:|---|
| train | 10,000 | Wikipedia `20231101.ja`, MeCab kana conversion |
| dev | 1,000 | Wikipedia `20231101.ja`, MeCab kana conversion |
| test | 2,000 | Wikipedia `20231101.ja`, MeCab kana conversion |

The splits are article-disjoint. The recorded manifest checks were:

- `article_disjoint_verified=true`
- `article_sha256_consistent=true`
- `contrast_pairs_disjoint=true`
- `split_field_verified=true`
- `span_tokens` is an integer in the range 1--3

The d1 files are externally supplied and are not present in this checkout.
Their individual SHA-256 values and the learned artifact SHA-256 were not
included in the supplied measurement record, so they are not fabricated here.
The available input hashes are:

| Input | SHA-256 |
|---|---|
| `data/dictionary_v2.tsv` | `1e423c314cb6ee117cb429cb7a9a063a199f4c3e80d9f95ffcb584f855566e0a` |
| `data/weights/decomposed_v1.json` | `029e82a617ed3e339a55be2cb85da486fb5777ac51442cb0a8d4faf0ff72c446` |

Wikipedia frequency and article distribution should not be treated as an
independent Japanese language distribution. Automatic MeCab kana conversion
can also introduce errors, especially for proper nouns, polyphonic words, and
numeric readings. AJIMEE, Mozc, and handmade data remain evaluation-only and
were not used for training or epoch selection.

## Training and Selection

Training used `decomposed_v1`, five epochs, learning rate `0.1`, and seed `9`.
Each epoch shuffled train indices with `random.Random(seed + epoch)`; dev and
test order was fixed. Epoch selection used only
`(end_to_end_error_rate, model_error_rate, epoch)` in lexicographic order.
The selected epoch was **4**. Test was evaluated only after this selection.

The supplied run record contains these train exact-match counts (coverage was
zero for every epoch):

| Epoch | Train exact |
|---:|---:|
| 1 | 6,770 |
| 2 | 6,752 |
| 3 | 6,694 |
| 4 | 6,699 |
| 5 | 6,665 |

Per-epoch dev metric values were stored in the run report but were not included
in the measurement values supplied for this documentation task; no values are
inferred here.

## Held-Out Test

| Metric | Baseline (`decomposed_v1`) | Learned averaged (epoch 4) |
|---|---:|---:|
| Coverage errors | 0 | 0 |
| Model errors | 620 (31.0%) | **538 (26.9%)** |
| End-to-end errors | 620 (31.0%) | **538 (26.9%)** |
| Exact matches | 1,380 (69.0%) | **1,462 (73.1%)** |

The improvement is a model-cost improvement, not a retrieval/coverage
improvement: all 2,000 test examples had a latent gold path.

## Artifacts and Separate-Process Check

The learned artifact was written to:

```text
artifacts/issue9-phase2/run1/weights.json
```

Its metadata recorded `averaged=true`, `updates`, `averaging_steps`,
`epochs=5`, `learning_rate=0.1`, and `seed=9`, together with dictionary,
initial-weight, and split provenance. The learned artifact hash was not part of
the supplied measurement record and the artifact is not present in this
checkout. It was successfully loaded in a separate process by:

```bash
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights artifacts/issue9-phase2/run1/weights.json --json
```

## Existing Evaluation Suites

| Suite | Baseline | Learned |
|---|---:|---:|
| AJIMEE CER | 0.290080 | **0.287092** |
| AJIMEE exact | 0.070 | **0.080** |
| Mozc | 88/498 | 87/498 |
| Handmade NDCG@5 | 0.992972 | 0.961061 |
| Handmade Top-1 | 14/14 (1.000) | 13/14 (0.928571) |

### Handmade Case Comparison

The baseline was correct on all 14 cases. The supplied record identifies one
regression:

| Case | Baseline | Learned | Result |
|---|---|---|---|
| `hashi-no` (`はしの`) | `橋の` | `橋野` (with `橋の` at rank 2) | regression |

The other 13 cases retained Top-1 according to the supplied record:
`kisha`, `hashi`, `ame`, `kaeru`, `koushou`, `kyoukai`, `seika`, `taishou`,
`kisha-wa`, `koushou-wo`, `kyoukai-de`, `koushou-history`, and
`koushou-business`. Several lower-ranked candidates changed, so aggregate
metrics alone should not be used as the regression explanation.

The likely cause of `hashi-no` is a Wikipedia-derived frequency/proper-noun
bias toward `橋野`, combined with a changed boundary preference for `の`.
This is a hypothesis, not a causal measurement.

## Limitations and Phase 3 Handoff

- Recalibrate confidence/cascade thresholds and reranker Viterbi blending after the learned cost scale changes.
- Compare lexical, transition, mapped-cost, katakana, source, and script ablations.
- Measure oracle recall@k separately from conditional ranking accuracy.
- Stratify by reading length, proper noun, numeric reading, and MeCab conversion quality.
- Audit Wikipedia frequency and article-disjoint provenance against evaluation leakage.
- Keep learned weights external or explicitly package their provenance and license metadata.
- Add case-level before/after diagnostics to future reports, including all per-query ranks and representative model errors.

## Verification

The supplied final verification reported:

```text
full pytest: 511 passed
```

The repository checkout used for this documentation task also passed the full
test suite before report creation. No d1 data, dictionary, existing weight, or
training artifact was generated or modified here.
