# Task 3: 読み条件付き頻度特徴

## 目的

issue #9 Phase 3 チェックリストの「読み条件付き頻度あり/なし」を検証できるようにする。設計ノート `docs/beme-learned-path-scorer.html` §4 の「読み内での表記頻度」に対応する特徴 `lex.reading_cond_logprob` と `lex.reading_candidate_count` を追加する。**外部コーパスは使わない。**

最初に `docs/missions/issue9-phase3/PLAN.md` の §3.3・§4（F0/F1 行）、`src/beme/features.py`、`src/beme/dictionary.py`、`src/beme/ipadic.py`、`scripts/apply_cost_learning.py` を読むこと。

## 依存

Task 1（特徴仕様レジストリと特徴フィルタ）が完了していること。

## 前提（調査済み・信じてよい）

- `data/corpus.txt` は **miniPC にも GPU マシンにも存在しない**。`.gitignore:5` で除外されている。
- GPU マシンの HuggingFace datasets キャッシュに `wikimedia/wikipedia` は無い（未取得）。
- したがって「コーパス頻度」を素直に数える経路は、100 MB の再取得を Phase 3 の律速にし、さらに「Wikipedia 頻度が評価データへリークしていないか」という監査対象（issue #9 Phase 3 注意点 5）を自分で増やす。
- `data/dictionary_v2.tsv` は 349,043 行。`raw_cost` 列を持つのは ipadic 由来の 348,910 行で、overlay 由来の 133 行は空。

## 対象ファイル

- 変更: `src/beme/features.py`（特徴の抽出と spec 追加）
- 変更: `src/beme/dictionary.py`（読み条件付き分布テーブルの構築）
- 変更: `scripts/build_ablation_weights.py`（F0/F1 条件の追加）
- 新規: `tests/test_reading_conditional.py`

`src/beme/lattice.py`・`converter.py`・`oracle.py`・`training.py`・`scorer.py`・辞書ファイルは変更しない。

## 変更内容

### 1. 特徴の定義

読み `r` を共有する辞書エントリ集合 `S(r)` 上で、IPADIC raw cost から条件付き分布を作る。

```text
p(s | r) = exp(-raw_cost(s, r) / T) / Σ_{s' ∈ S(r)} exp(-raw_cost(s', r) / T)

lex.reading_cond_logprob    = -log p(s | r)
lex.reading_candidate_count = log(1 + |S(r)|)
```

- `T` は辞書構築時の raw cost スケールに合わせた定数。**`beme.ipadic` の既存定数から決めること。新しいマジックナンバーを導入しない。** 既存定数で表せない場合は、`features.py` ではなく `beme.ipadic` 側に公開定数として 1 箇所に定義し、由来をコメントに書く。
- `raw_cost` を持たない行（overlay 133 行・unknown・literal）は、`|S(r)|` の数え上げには含めるが `lex.reading_cond_logprob` を立てない。
- 数値安定性のため、正規化は max を引く log-sum-exp で計算すること。`exp` を直接足すと `raw_cost` の範囲（数千オーダー）でアンダーフローする。

### 2. なぜこれが `lex.ipadic_cost_mapped` と別物か（実装時に混同しないこと）

`lex.ipadic_cost_mapped` は単語の**絶対**コストである。`lex.reading_cond_logprob` は「その読みの候補集合の中での相対的な強さ」であり、**正規化**が新しい情報を運ぶ。読みの曖昧度が高いほど値が伸びる。この差が出ることを、曖昧な読み（例 `こうしょう`）と一意な読みの両方でテストに固定すること。

この 2 つは相関するので、アブレーション F1（`+freq_v1-no_ipadic_cost`）が「頻度が word cost を代替できるか」を測る条件になる。

### 3. テーブルの構築場所

`Dictionary` のロード時に 1 回だけ構築する。

- 349,043 行に対する 1 パスで、読みごとの `logsumexp` と件数を集める。
- テーブルは読み文字列 → `(log_partition, count)` の dict。
- **ロード時間と RSS の増分を測って報告すること。** Phase 1 のゲート（ロード 2.0x / RSS 1.6x 以内）を引き継ぐ。現行 `dictionary_v2.tsv` のロードは Phase 1 実測で v1 比 1.25x / RSS 1.30x。
- `FeatureExtractor` は `Dictionary`（またはテーブル）への参照を持つ必要がある。**`FeatureExtractor.__init__` のシグネチャを変える場合、既存の全呼び出し箇所（`features.py` の `legacy_v1_extractor` / `decomposed_v1_extractor`、`scripts/train_scorer.py`、`scripts/evaluate.py`、Task 1 のレジストリ）を洗い出して更新すること。** テーブル未設定のときにこの特徴を要求する spec を作ったら、黙って 0 にせず例外にする。

### 4. キャッシュ前提の確認

`LinearScorer._edge_cache` は `edge.token` をキーにする（`scorer.py:51`）。この特徴は `token.reading` と `token.raw_cost` だけから決まる（辞書テーブルは固定）ので、位置非依存の前提は保たれる。`edge.start` / `edge.end` を参照しないこと。

### 5. spec の追加

Task 1 のレジストリへ次を追加する。

| spec id | 内容 |
|---|---|
| `decomposed_v1+freq_v1` | `decomposed_v1` + `lex.reading_cond_logprob` + `lex.reading_candidate_count` |
| `decomposed_v1+freq_v1-no_ipadic_cost` | 上記から `lex.ipadic_cost_mapped` を除去 |

Task 2 が完了していれば `decomposed_v1+matrix_v1+freq_v1`（PLAN §4 の X0）も追加する。Task 2 未完なら、X0 の追加は Task 8 の担当者へ引き継ぐ旨を報告に書く。

初期重みは新規 2 特徴とも **0.0** で始める。

## 必須テスト

- `legacy_v1` / `decomposed_v1` の出力が本タスク前と完全一致（既存 golden）。
- 一意な読み（`|S(r)| == 1`）で `lex.reading_cond_logprob == 0.0`（`p == 1`）かつ `lex.reading_candidate_count == log(2)`。
- 曖昧な読みで、raw_cost が小さいエントリほど `lex.reading_cond_logprob` が小さい（順序が保たれる）。
- 同じ読みのエントリ全体で `Σ exp(-lex.reading_cond_logprob) == 1.0`（浮動小数の許容誤差内）。
- raw_cost を持たない token（overlay / unknown / literal）で `lex.reading_cond_logprob` が立たない。
- 数千オーダーの raw_cost でアンダーフロー・オーバーフローしない（極端値の合成辞書で固定）。
- `lex.reading_cond_logprob` が `lex.ipadic_cost_mapped` の定数倍でないこと（曖昧度の異なる 2 つの読みで比が変わることを示す）。
- テーブル未設定で当該特徴を要求する spec が例外になる。
- 辞書ロード時間と RSS の増分が Phase 1 のゲート内。

## 完了条件

- `decomposed_v1` の既存挙動が完全に不変。
- `+freq_v1` で学習と評価の両方が動く。
- 外部コーパス・ネットワーク・MeCab を一切必要としない。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_reading_conditional.py
uv run pytest -q tests/test_features.py tests/test_dictionary_lattice.py tests/test_scorer_golden.py tests/test_decomposed_scorer.py
uv run pytest -q

# 既存挙動の不変（期待値: NDCG@5=0.992972 / CER=0.290080 / Mozc 88/498）
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json

uv run python scripts/build_ablation_weights.py
uv run python scripts/evaluate.py --suite handmade --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/ablations/decomposed_v1+freq_v1.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- **外部コーパスを取得しない。** `scripts/fetch_corpus.py` を実行しない。Wikipedia 由来のコーパス頻度版（PLAN §3.3 の F2）は Phase 3 の既定スコープ外であり、本タスクの対象ではない。
- 対象外ファイルを変更しない。`FeatureExtractor` のシグネチャを変える場合のみ、影響箇所の更新を許す（範囲を報告すること）。
- 位置依存特徴を追加しない。`edge.start` / `edge.end` を参照しない。
- 新しいマジックナンバーを導入しない。
- 辞書ファイルを生成・変更しない。
- テストを実装と同時に追加し、上記コマンドの実結果（ロード時間・RSS を含む）を報告する。
