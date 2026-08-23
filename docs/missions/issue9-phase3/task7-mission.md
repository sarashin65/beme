# Task 7: コスト尺度の再較正測定 CLI

## 目的

issue #9 Phase 3 注意点 1「学習でコストスケールが変わるため、confidence / cascade 閾値とリランカの Viterbi ブレンド比の再較正が必要」に対応する測定を実装する。

**ただし、コードを読んだ結果、注意点 1 の前提の一部が現状と食い違うことが判明している。** 下記の「調査で判明した事実」を読んでから設計すること。あなたの仕事は閾値を「直す」ことではなく、**再較正に必要な統計量を測り、現状の設計上の欠落を正確に記録すること**である。

最初に `docs/missions/issue9-phase3/PLAN.md` の §2.2(e)(f)・§3.6、`src/beme/converter.py`（特に 164-201 行）、`src/beme/evaluation.py`（220-290 行）、`src/beme/benchmark.py`、`src/beme/reranker.py` を読むこと。

## 依存

なし。Task 1〜6 と並行して進めてよい。

## 調査で判明した事実（実測済み・信じてよい）

### (a) cascade 閾値は現状どの精度指標にも効いていない

`src/beme/converter.py:174` が `needs_rerank = confidence is not None and confidence < threshold` を計算しているが、実際のリランク発火は次行の

```python
reranked = not context.is_empty and self.reranker is not None
```

であり、**`needs_rerank` はどこからも参照されていない**（`src/beme/` と `scripts/` 全体を grep 済み）。`evaluation.py:255` が `rerank_rate` として集計するだけである。

したがって `--cascade-threshold` を変えても AJIMEE CER / Mozc pass / handmade NDCG・Top-1 は 1 つも動かない。**閾値を精度で最適化してはならない**（動かない指標を最適化することになる）。

### (b) `viterbi_weight` は mock リランカでは使われない

`viterbi_weight * candidate.cost` を使うのは `CrossEncoderReranker`（`reranker.py:293`）と `LLMLogitsReranker`（`reranker.py:433`）だけで、既定の `MockContextReranker` は候補コストを混ぜない。実モデル（LLM gguf・cross-encoder）は miniPC の `~/.cache/beme/` にのみキャッシュされており、GPU マシンには無い。

「局所モデルの K-best をリランカへ渡す」こと自体が **Phase 4 のチェックリスト項目**である。よって **`viterbi_weight` の実較正は Phase 3 のスコープ外**とし、本タスクは Phase 4 が初期値に使える倍率を 1 つ出すところまでとする。

### (c) confidence を持つのは handmade の per-query だけ

`evaluation.py:266` の per_query には `confidence` があるが、AJIMEE の `per_case`（`benchmark.py:215-227`）と Mozc の `per_case`（`:327-339`）には無い。**既存評価の出力形式を変えずに**、独立したスクリプトで `Converter.convert` を直接呼んで収集すること。

## 対象ファイル

- 新規: `scripts/measure_cost_scale.py`
- 新規: `tests/test_measure_cost_scale.py`

`src/beme/` 配下は変更しない。**`needs_rerank` を実際のゲートとして配線する修正をしてはならない**（変換挙動を変えるとアブレーション結果の解釈が汚れる。Phase 4 の設計判断として引き継ぐ）。

## 変更内容

### 1. confidence 分布の測定

`--baseline-weights` と `--learned-weights` の 2 条件で、次のデータセットについて `confidence = candidates[1].cost - candidates[0].cost`（`converter.py:173`）の分布を出す。

- handmade 14 / Mozc 498（`--suite`）
- d1 split（`--data`・GPU マシンでのみ実行可能）
- AJIMEE 200 は `convert_sentence` で文分割されるため、**文ではなく分割後の各読みごと**の confidence を集める。定義が違うので別セクションに出し、混ぜないこと。

出力する統計: 件数、候補が 1 個以下で confidence が `null` になった件数、p10 / p25 / p50 / p75 / p90 / 平均 / 最小 / 最大。

### 2. cascade 閾値の再較正値

判定基準は **「baseline の `cascade_threshold=2.0` と同じ発火率を与える learned 側の閾値」**とする。精度で選ばない（事実 (a) のため動かない）。

```text
rate_baseline = P(confidence_baseline < 2.0)
threshold_learned = quantile(confidence_learned, rate_baseline)
```

データセットごとに算出し、**データセット間で値が食い違う場合は 1 つに丸めず全部出す**。丸めるのは Phase 4 の判断。

### 3. Viterbi ブレンド比の初期倍率

`viterbi_weight * cost` の項が baseline と learned で何倍スケールが変わったかを、**候補間コスト差の分布比**として 1 つの数値で出す。

```text
scale_ratio = median(confidence_learned) / median(confidence_baseline)
```

Phase 4 の出発点は `viterbi_weight = 1.0 / scale_ratio` になる。この式と根拠を出力 JSON の中に文字列として埋めること（レポート作成者が導出を再構成できるように）。

**中央値以外に p25 / p75 の比も出すこと。** 分布の形が変わっていれば単一倍率で吸収できないので、その判断材料になる。

### 4. 設計上の欠落の記録

出力 JSON に次を機械可読な形で必ず含める。

```json
"design_findings": {
  "needs_rerank_is_unused": {
    "confirmed": true,
    "detail": "converter.py:174 computes needs_rerank; converter.py:176 gates reranking on context/reranker presence only. No caller reads needs_rerank except evaluation.py:255 (rerank_rate aggregation).",
    "impact": "cascade_threshold does not affect any accuracy metric; recalibration is a statistic, not an optimization."
  },
  "viterbi_weight_out_of_scope": {
    "reason": "used only by CrossEncoderReranker and LLMLogitsReranker; real models are cached on the miniPC only; wiring K-best into a reranker is a Phase 4 checklist item."
  }
}
```

`confirmed` は**実際に検証した結果を書くこと**。あなた自身が grep して確かめ、事実と違えばそう書く。上記の記述をそのままコピーしない。

### 5. CLI

```text
--data / --suite       対象データ（--suite は {handmade, mozc, ajimee}）
--dictionary           既定 data/dictionary_v2.tsv
--baseline-weights     必須
--learned-weights      必須
--baseline-threshold   既定 2.0（再較正の基準となる現行値）
--top-k                既定 5
--beam-width           既定 64
--output               JSON 出力先（必須）
--json                 標準出力にも出す
```

決定性: UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行、atomic replace、2 回実行で byte-identical。分位点の計算方法（線形補間の有無）を出力に明記すること。

## 必須テスト

- 既知の confidence 列に対する分位点が手計算値と一致（分位点の定義を固定する）。
- 候補が 1 個のケースで `confidence` が `null` になり、分位点計算の母数から除かれ、その件数が別途出る。
- 再較正閾値が「同じ発火率」を与えることの検算（learned 側で `P(conf < threshold_learned)` が `rate_baseline` と一致）。
- baseline と learned に同じ重みを渡すと `scale_ratio == 1.0` かつ `threshold_learned == 2.0`。
- AJIMEE が読み単位で集計され、handmade / Mozc と別セクションに出る。
- `design_findings` が実際の grep 結果に基づく（`needs_rerank` の参照箇所を列挙するテスト）。
- 2 回実行が byte-identical。

## 完了条件

- 4 データセットの confidence 分布が baseline / learned で出る。
- 再較正閾値と Viterbi 初期倍率が算出される。
- `needs_rerank` が未使用である事実が機械可読な形で記録される。
- **変換の挙動を 1 つも変えていない**（既存 4 指標が不変）。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_measure_cost_scale.py
uv run python scripts/measure_cost_scale.py --help
uv run pytest -q

# 既存挙動が不変（期待値: NDCG@5=0.992972 / CER=0.290080 / Mozc 88/498）
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json

# 同一重みでの自己整合（scale_ratio=1.0 / threshold=2.0 になるはず）
uv run python scripts/measure_cost_scale.py --suite handmade \
  --dictionary data/dictionary_v2.tsv \
  --baseline-weights data/weights/decomposed_v1.json \
  --learned-weights data/weights/decomposed_v1.json \
  --output /tmp/cost_scale_selfcheck.json --json

uv run python scripts/measure_cost_scale.py --suite mozc \
  --dictionary data/dictionary_v2.tsv \
  --baseline-weights data/weights/decomposed_v1.json \
  --learned-weights data/weights/decomposed_v1.json \
  --output /tmp/cost_scale_mozc.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `src/beme/` 配下を変更しない。**`needs_rerank` を配線しない。** `benchmark.py` / `evaluation.py` の出力形式を変えない。
- 閾値を精度指標で最適化しない（事実 (a) により動かないため、最適化したように見えても偶然のノイズ）。
- 実リランカ（LLM / cross-encoder）を呼ばない。ネットワーク・GPU・外部モデルを必要としない CLI/テストにする。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。
- `design_findings` は自分で検証した結果を書く。指示書の文言をコピーしない。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
