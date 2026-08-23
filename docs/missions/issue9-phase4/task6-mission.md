# Task 6: cascade ゲート配線とレイテンシ予算（**条件付き**）

## 実施条件

**Task 5 の結果が PLAN §5 の「効いた」分岐に該当する場合のみ実施する。**

> contrast pair 128 件で LR が L に対し McNemar p < 0.05 かつ改善方向、**かつ 3 seed すべてで同じ向き**

「効いたが不安定」「効かなかった」に該当する場合、**このタスクは実施しない。** ゲートを配線する理由が無い。その場合は Task 7 のレポートで、issue コメントの 3 つの切り分け（結合比 / 識別力 / 文脈長）を書いて #2・#3 へ引き継ぐ。

**判定は実測で行う。Task 5 のパイロット的な数字や PLAN §2.4 の 40 件パイロットを根拠に先取りしないこと。**

## 目的

cross-encoder は 1 変換あたり約 3.0 秒（miniPC・N100・ユニーク候補 47.9 × 63 ms）かかる。**全候補に回す設計は IME として成立しない。** リランカが効くと分かった場合、効くケースにだけ適用する配線を入れる。

`src/beme/converter.py:174` の `needs_rerank`（`confidence < cascade_threshold`）は計算されるだけで**何もゲートしていない**。実際の発火は `:176` の `not context.is_empty and self.reranker is not None` だけで決まり、`needs_rerank` の参照は `evaluation.py:250` の集計のみである。

最初に `docs/missions/issue9-phase4/PLAN.md` の §3.6 と §2.6、`src/beme/converter.py:130-204`、`src/beme/evaluation.py:230-280`、Task 5 の C5 出力（ゲート候補の recall / fire_rate 曲線）を読むこと。

## 依存

Task 5。C5 の出力が入力になる。

## 対象ファイル

- 変更: `src/beme/converter.py`（`needs_rerank` をゲートとして配線）
- 変更: `scripts/evaluate.py` / `scripts/rerank_eval.py`（ゲートの有効化フラグ）
- 新規: `tests/test_cascade_gate.py`

## 設計上の前提（実測済み・推測ではない）

- 削減手段は 2 つあり、効果と副作用が違う。**分けて測ること。**

  | 手段 | 削減率 | 副作用 |
  |---|---|---|
  | `rerank_k` を絞る | 候補数に比例（`rerank_k=10` なら 47.9 → 10 で約 4.8 倍・約 630 ms） | `k > rerank_k` の gold が救えない |
  | cascade ゲート | 発火率に比例 | **ゲートの recall がそのまま改善の上限になる** |

- **confidence 単独ではゲートにならない見込みが高い。** PLAN §2.6 のとおり、Phase 3 実測で learned confidence の p75 = 2.45 なので「conf < 2.0」の発火率は約 .72。パイロット 40 件の contrast pair は全件が conf < 0.5 だったが、conf < 0.5 は全体の 6 割強を占める。**「confidence 単独では足りない」という結論も正式な成果**である。無理に閾値をひねり出さないこと。

## やること

### 1. ゲートの評価（精度ではなく recall / fire_rate で）

Task 5 の C2 で「LR が L から救った c 件」が確定している。ゲート条件 `g` に対して

```text
recall(g)    = |{c 件のうち g が発火}| / |c 件|
fire_rate(g) = |{全例のうち g が発火}| / 全例
```

を出し、**`recall ≥ 0.95` を満たす中で `fire_rate` が最小の `g`** を選ぶ。閾値は Task 5 の C5 曲線から選び、**選択は d1 dev で行い test で 1 回確認する**（Task 5 と同じ規律）。

ゲート候補（最低この 3 つを比較する）:

1. `confidence < θ`（Phase 3 の再較正値 `2.025162545128425` を出発点）
2. `confidence < θ` ∧ 上位 2 候補の表層が同じ読み長
3. 上位 m 候補のコスト差に基づく条件（m は C5 の gold_rank 分布から決める）

**`recall ≥ 0.95` を満たす候補が 1 つも無い、または満たす候補の `fire_rate` が 0.5 を超える場合、「ゲートでは削減できない」と結論して配線を入れない。** その場合は `rerank_k` の削減だけをレポートし、レイテンシ要件を満たせない事実を明記する。

### 2. 配線

`converter.py` で `needs_rerank` を実際のゲートにする。**既定の挙動を変えないこと**が絶対条件。

```text
reranked = not context.is_empty and self.reranker is not None
if self.cascade_gate_enabled:
    reranked = reranked and needs_rerank
```

- ゲートは**既定で無効**（`cascade_gate_enabled=False`）。有効化は明示的な引数・CLI フラグでのみ。
- `ConversionResult` の `needs_rerank` / `reranked` の意味が変わるので、`to_dict()` の契約とテストを更新する。**`reranked` は「実際にリランクしたか」であり続けること。**

### 3. レイテンシ予算

次を表にする。**すべて miniPC（Intel N100・4 core）実測で、実行機名を併記する。**

| 設定 | 発火率 | 1 変換あたり（発火時） | 1 変換あたり（期待値） | contrast 128 の Top-1 |
|---|---:|---:|---:|---:|
| 全候補・ゲート無し | 1.00 | 約 3.0 s | 約 3.0 s | Task 5 の LR |
| `rerank_k` 選択値・ゲート無し | 1.00 | | | |
| `rerank_k` 選択値・ゲート有り | | | | |

**目標レイテンシを勝手に決めないこと。** IME として許容できる値はユーザーの判断であり、このタスクは「この設定なら何ミリ秒で Top-1 がいくつ」という対応表を出すところまでである。

## 必須テスト

1. `cascade_gate_enabled=False`（既定）のとき、`convert` の出力が変更前と**完全一致**する。
2. `cascade_gate_enabled=True` かつ `needs_rerank=False` のとき、リランカが**呼ばれない**（呼び出し回数を数える test double で固定）。
3. `cascade_gate_enabled=True` かつ `needs_rerank=True` のとき、従来どおりリランクされる。
4. `context.is_empty` のときはゲートに関わらずリランクされない。
5. 候補が 1 件以下で `confidence is None` のときのゲート挙動を決めてテストで固定する（**`None` を「発火する」と「発火しない」のどちらに倒したか、判断を報告すること**）。
6. `ConversionResult.to_dict()` の `needs_rerank` / `reranked` の意味がテストで固定されている。
7. 既存の全テストが通る。

## 完了条件

- ゲートが `recall / fire_rate` で評価され、選択根拠が数値で示されている。
- 既定の挙動が 1 bit も変わっていない（下記の不変条件コマンドで確認）。
- レイテンシ予算の対応表が miniPC 実測で出ている。
- 「ゲートでは削減できない」という結論に至った場合、**それを結論として書き、配線を入れていない**。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_cascade_gate.py
uv run pytest -q

# 不変条件（既定でゲート無効。Phase 3 と bit 一致）
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json --json
# 期待: Handmade NDCG@5 0.992972 / Top-1 1.000000、AJIMEE CER 0.290080 / exact 0.070000、Mozc 88/498

# ゲート有効時のレイテンシ（miniPC）
uv run python scripts/rerank_eval.py score \
  --data <test の一部> --reranker cross-encoder --cross-encoder-local-files-only \
  --torch-threads 4 <ゲート有効フラグ> --output /tmp/gated.jsonl
```

## 報告に含めること

- 上記コマンドの**実出力**。
- ゲート候補 3 種以上の `recall` / `fire_rate` の表。
- 選んだゲート条件と、選ばなかった条件を落とした理由。
- `confidence is None` の扱いをどちらに倒したか、その判断。
- レイテンシ予算の対応表（**実行機名併記**）。
- 「ゲートでは削減できない」に至った場合はその判定根拠。

## 共通ルール・制約

- `git commit` / `git push` 禁止。
- **既定の挙動を変えない。** ゲートは opt-in。
- 目標レイテンシを勝手に決めない。対応表を出すところまで。
- 選択は d1 dev、確認は test 1 回。Task 5 と同じ規律。
- `recall ≥ 0.95` を満たす候補が無いなら、閾値をひねり出さずに「できない」と書く。
