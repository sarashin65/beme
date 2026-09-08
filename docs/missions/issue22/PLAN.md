# issue #22: C0 + IPADIC matrix 固定重み実験計画

## 1. 目的と仮説

### 目的

現行ベース C0（`decomposed_v1-no_script-surface_len+corpus_v1`）へ、Phase 3 で実装済みの `trans.matrix_cost` を追加する。matrix の重みは学習せず、MeCab の word cost と connection cost の相対スケールに対応する `w_matrix = 0.5` に固定する。AJIMEE-Bench `JWTD_v2/v1` 全 200 件で C0 と同一条件の paired 比較を行い、IPADIC の細粒度 left/right ID に基づく接続コストが長文の分割誤りを減らすかを検証する。

この実験は重み探索ではない。`0.5` は評価結果を見る前に固定し、AJIMEE 上で再調整しない。d1 test は 86.5% が単一 token で transition を十分観測できないため、主評価には用いない。

### 仮説

- **主仮説 H1**: C0+matrix は C0 より AJIMEE の分割誤り率を低下させる。
- **帰無仮説 H0**: paired case 単位で、C0 の分割誤りが解消する確率と、C0+matrix により分割誤りが新たに発生する確率は等しい。
- **副仮説**: 分割誤りの低下は CER と完全一致率を悪化させず、aggregate な変換品質の改善または維持につながる。

期待する機構は、16×16 の手書き POS 遷移表では失われた細粒度の接続選好を `left_id/right_id` から回収し、内容語内部の過分割や境界ずれを抑えることである。一方、matrix は同音異表記の選択や辞書 coverage を直接は改善しないため、CER・完全一致の効果は分割誤り率より小さい可能性がある。

## 2. 比較条件と固定条件

| 条件 | weights | 差分 |
|---|---|---|
| C0（対照） | `data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json` | 現行の coarse POS transition |
| C0+matrix（処置） | `data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json` | C0 と同じ重み + `trans.matrix_cost: 0.5` |

両条件で次を固定する。

- dictionary: `data/dictionary_v2.tsv`
- benchmark: `data/ajimee_bench_jwtd_v2_v1.json` 全 200 件（除外なし）
- `top_k=5`, `beam_width=64`, `cascade_threshold=2.0`
- reranker: `mock`
- AJIMEE の `context_text`、カタカナ→ひらがな正規化、複数許容表記の扱いは既存 `scripts/evaluate.py` / `beme.benchmark.evaluate_ajimee` に従う
- 学習、seed sweep、matrix 重み sweepは行わない

唯一の操作変数は `trans.matrix_cost` の有無である。matrix 条件でも既存の coarse POS transition を残すため、「全面 matrix 置換」ではなく C0 への追加信号を評価する。

## 3. 準備物の確認

2026-09-08 時点で、指定された 4 ファイルの存在を確認済みである。

| ファイル | 確認内容 |
|---|---|
| `src/beme/features.py` | `CORPUS_MATRIX_V1_A8_SPEC` が registry に登録され、transition state は `(pos, right_id)` |
| matrix weights JSON | spec 名が指定どおりで、`trans.matrix_cost == 0.5` |
| `data/ipadic_matrix_v1.bin` | 存在、1,289,632 bytes。`beme.matrix_cost.DEFAULT_MATRIX_PATH` と一致 |
| AJIMEE JSON | 存在、132,231 bytes、200 件 |

実行開始時にも、取り違え防止のため以下を実行する。

```bash
test -f src/beme/features.py
test -f data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json
test -f data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json
test -f data/ipadic_matrix_v1.bin
test -f data/ajimee_bench_jwtd_v2_v1.json

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from beme.benchmark import load_ajimee_cases
from beme.features import registered_spec_ids
from beme.matrix_cost import DEFAULT_MATRIX_PATH, MatrixCostTable

base_path = Path("data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json")
matrix_path = Path("data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json")
base = json.loads(base_path.read_text())
matrix = json.loads(matrix_path.read_text())
spec = "decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1"

assert spec in registered_spec_ids()
assert matrix["feature_spec"] == spec
assert matrix["weights"]["trans.matrix_cost"] == 0.5
assert {k: v for k, v in matrix["weights"].items() if k != "trans.matrix_cost"} == base["weights"]
assert DEFAULT_MATRIX_PATH.resolve() == Path("data/ipadic_matrix_v1.bin").resolve()
assert len(load_ajimee_cases("data/ajimee_bench_jwtd_v2_v1.json")) == 200
table = MatrixCostTable.load(DEFAULT_MATRIX_PATH)
print({"ajimee_cases": 200, "matrix_ids": len(table.ids), "w_matrix": 0.5})
PY

git rev-parse HEAD
python --version
sha256sum \
  data/dictionary_v2.tsv \
  data/ajimee_bench_jwtd_v2_v1.json \
  data/ipadic_matrix_v1.bin \
  data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json \
  data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json
```

確認済み SHA-256 のうち、本実験で特に固定する値は次のとおり。

- AJIMEE: `e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca`
- matrix binary: `96eb8b1359a4382bdf3962a656cafbf1d687cb4d4a08a570d388db49d335646e`
- C0 weights: `e05e30fc459aa07c59f6dd0855f40007ecdbfb67b997126a197766c57853e6a9`
- C0+matrix weights: `22039548c9cdc0ac2bf7a86d32f24c6584bf6977742710c502f46a36052e0401`

いずれかの assertion または hash が一致しなければ評価を止め、差分を記録してから計画を改訂する。

## 4. 実行手順

### 4.1 出力先の作成と AJIMEE 評価

```bash
mkdir -p artifacts/issue22-matrix-fixed

python scripts/evaluate.py \
  --suite ajimee \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json \
  --ajimee-cases data/ajimee_bench_jwtd_v2_v1.json \
  --top-k 5 \
  --beam-width 64 \
  --cascade-threshold 2.0 \
  --reranker mock \
  --json \
  > artifacts/issue22-matrix-fixed/c0-ajimee.json

python scripts/evaluate.py \
  --suite ajimee \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json \
  --ajimee-cases data/ajimee_bench_jwtd_v2_v1.json \
  --top-k 5 \
  --beam-width 64 \
  --cascade-threshold 2.0 \
  --reranker mock \
  --json \
  > artifacts/issue22-matrix-fixed/c0-matrix-ajimee.json
```

両 JSON の `case_count == 200`、ID 集合の完全一致、`k == 5` を assertion で確認する。決定性の smoke check として各条件をもう一度別ファイルへ出力し、`cmp` で一致を確認する。計時フィールド等で JSON が非決定的な場合は `per_case` の `id/hypothesis/cer/exact_match` のみ比較する。

### 4.2 分割誤りの paired 分析

`scripts/evaluate.py` は CER と完全一致を出すが、token 境界を出さない。このため分割分析では、既存 `scripts/c0_ajimee_deepdive.py` と同じ `build_lattice` + `constrained_best_path` の方法を使う。ただし同スクリプトは比較パスが A8/C0 に hard-code され、`mean_cer` に平均編集距離を入れているため、**そのままの集計値は使わない**。実験実行タスクでは、このロジックをパラメータ化した paired analyzer を用意し、次の CLI 契約で実行する。

```bash
python scripts/compare_ajimee_segmentation.py \
  --dictionary data/dictionary_v2.tsv \
  --cases data/ajimee_bench_jwtd_v2_v1.json \
  --baseline-weights data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1.json \
  --treatment-weights data/weights/ablations/decomposed_v1-no_script-surface_len+corpus_v1+matrix_v1.json \
  --top-k 5 \
  --beam-width 64 \
  --cascade-threshold 2.0 \
  --output artifacts/issue22-matrix-fixed/paired-segmentation.json
```

この analyzer は production 実装を変更せず、評価専用とする。最低限、各 case について `id`, 両条件の `hypothesis`, `cer`, `exact_match`, `gold_in_lattice`, `error_type`, `gold_boundaries`, `hypothesis_boundaries` を保存する。

分割判定は次の順序で固定する。

1. `katakana_to_hiragana` 後の同一 reading から両条件の Top-1 を生成する。
2. 完全一致なら `correct` とする。
3. 不正解の場合、許容 reference ごとに `constrained_best_path` を求める。評価時の Top-1 に対して `(CER, edit distance, reference の元の順序)` が最小の reference を決定的に選ぶ。
4. reference path が存在しなければ `coverage` とする。
5. reference path と hypothesis path の reading 累積位置（末尾を除く内部境界集合）を比較する。一致なら `homophone`、不一致なら `segmentation` とする。
6. 両条件で同じ case ID 200 件が揃わない場合は集計しない。

複数許容 reference の最初だけを常に gold とする方式は採用しない。CER が評価で選んだ許容表記と分割判定の gold がずれることを避けるためである。

### 4.3 paired 統計と結果表

paired analyzer の出力から以下を計算し、`artifacts/issue22-matrix-fixed/statistics.json` に保存する。

- 分割誤りの 2×2 表: `both_segmentation`, `resolved`（C0 のみ分割誤り）, `induced`（matrix のみ分割誤り）, `neither`
- `resolved` と `induced` に対する two-sided McNemar 検定。既存 `beme.rerank_stats.mcnemar` に従い、discordant case が 25 未満なら exact binomial、25 以上なら continuity-corrected chi-square を使用する
- case を単位とする paired bootstrap 10,000 回、seed `20260908`
  - `CER(C0+matrix) - CER(C0)` の平均差 95% percentile CI
  - `exact(C0+matrix) - exact(C0)` の率差 95% percentile CI
  - `segmentation_error(C0+matrix) - segmentation_error(C0)` の率差 95% percentile CI
- CER 改善／同値／悪化件数、exact の fixed／regressed 件数
- error type の遷移表（特に `segmentation -> homophone/correct` と `homophone/correct -> segmentation`）

`mcnemar` には `b=resolved`, `c=induced` を渡す。この関数の通常の correctness 比較とは向きが逆になるため、変数名だけで解釈せず、出力 JSON に定義を明記する。

## 5. 評価指標

### 主指標: 分割誤り率

全 200 件に対する `error_type == segmentation` の割合を主指標とする。

$$
\mathrm{SegErrorRate}=\frac{N_{\mathrm{segmentation}}}{200}
$$

併せて、coverage を除くモデル誤り内の構成比

$$
\frac{N_{\mathrm{segmentation}}}{N_{\mathrm{segmentation}}+N_{\mathrm{homophone}}}
$$

も記載するが、分母が条件で変わるため合否判定には使わない。過分割／過連結／境界ずれの内訳も、内部境界の差集合から補助的に集計する。

### 副指標

- **Mean CER**: 各 case で許容 reference のうち最小 CER を選び、その macro mean を取る。低いほど良い。
- **完全一致率**: Top-1 が許容 reference のいずれかと完全一致した case 数 / 200。高いほど良い。
- **coverage error 数**: 分割誤りが coverage へ移っただけでないことを確認する guardrail。

`scripts/c0_ajimee_deepdive.py` の既存 `mean_cer` は実際には平均編集距離なので、CER の公式値は必ず `scripts/evaluate.py` の `mean_cer` を採用する。

## 6. 合格基準

評価前に次を事前登録する。主検定は 1 本だけなので多重比較補正は行わない。CER、完全一致、error subtype は副次解析とする。

### Pass

次の全条件を満たす場合、matrix は「分割誤りを有意に減らした」と判定する。

1. `SegErrorRate(C0+matrix) < SegErrorRate(C0)`。
2. `resolved > induced` かつ two-sided McNemar `p < 0.05`。
3. segmentation error rate の paired bootstrap 95% CI の上限が 0 未満。
4. guardrail として coverage error 数が C0 より増えず、mean CER と完全一致率の点推定がいずれも悪化しない。

### Fail / Inconclusive

- 分割誤りが同数または増加した場合は **Fail**。
- 分割誤りの点推定は低下したが有意差の条件を満たさない場合は **Inconclusive（効果未確認）**。
- 分割誤りは有意に減るが CER、完全一致、coverage の guardrail を破る場合は **Inconclusive / trade-off** とし、そのまま採用しない。
- 200 件で discordant case が少なく検出力不足の場合も、有意でない結果を「効果なし」と断定せず **Inconclusive** と記録する。ただし AJIMEE 上で追加の weight tuning は行わない。

合否と別に、効果量として分割誤りの絶対差（件数、percentage points）、相対減少率、`resolved/induced` を必ず併記する。p 値だけで採否を決めない。

## 7. 報告物

実験完了時に次を保存する。

- `artifacts/issue22-matrix-fixed/c0-ajimee.json`
- `artifacts/issue22-matrix-fixed/c0-matrix-ajimee.json`
- `artifacts/issue22-matrix-fixed/paired-segmentation.json`
- `artifacts/issue22-matrix-fixed/statistics.json`
- commit、Python version、入力 SHA-256、全コマンド、判定を含む結果レポート

最終表は最低限、以下の形にする。

| 指標 | C0 | C0+matrix | 差（matrix − C0） | 95% CI / p | 判定 |
|---|---:|---:|---:|---:|---|
| Mean CER |  |  |  | paired bootstrap CI |  |
| 完全一致 |  |  |  | paired bootstrap CI |  |
| 分割誤り |  |  |  | paired bootstrap CI / McNemar p |  |
| Coverage error |  |  |  | — | guardrail |

## 8. 次ステップへの示唆

- **Pass かつ品質 guardrail も通過**: `w_matrix=0.5` を C0 の採用候補とし、未使用の AJIMEE 外データまたは新規 held-out 長文で再現性を確認する。続いて handmade/Mozc の回帰、レイテンシ、matrix binary の配布サイズを gate にする。
- **分割は改善するが CER/完全一致が悪化**: matrix が境界を直しても表記順位を壊している。`w_matrix` を同じ AJIMEE で sweep せず、独立 dev set を用意して重み較正を別実験として事前登録する。
- **有意差なし**: d1 の単一 token 偏重だけが原因ではなく、固定 `0.5` でも matrix 信号の追加効果が小さい可能性が高い。`resolved/induced` と過分割／過連結の方向別内訳を調べ、overlay / 一文字 fragment の lexical calibration を優先する。
- **coverage が増える**: 分割誤りが別カテゴリへ移動しただけなので不採用とし、left/right ID 欠損、unknown/literal/BOS 周辺の matrix lookup を診断する。
- **特定 subtype のみ改善**: 改善した left/right ID pair と悪化 pair を抽出し、全面追加ではなく限定的な transition feature 化を独立仮説として検討する。

いずれの場合も、この 200 件の結果を見て `0.5` を変更し、同じ AJIMEE で再評価した値を確認実験として報告してはならない。探索が必要なら dev/evaluation を分離した新しい実験にする。
