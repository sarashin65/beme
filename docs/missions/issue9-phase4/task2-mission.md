# Task 2: リランカ評価ハーネス（score / combine の 2 段分離）

## 目的

d1 の `left_context` / `right_context` を実リランカへ渡し、Top-1 を測る導線を作る。`scripts/evaluate.py` は suite 単位で d1 を扱えず、`scripts/diagnose_cases.py` はリランカを通さない。

**このタスクの中核は 2 段分離である。** cross-encoder の relevance は `(context, surface)` だけの関数であり、`viterbi_weight` にも `rerank_k` にも依存しない（`src/beme/reranker.py:275-301`）。したがって relevance を 1 回計算してキャッシュすれば、`viterbi_weight` × `rerank_k` × ゲート閾値の全グリッドが**追加の推論ゼロで厳密に**評価できる。これは近似ではない。

最初に `docs/missions/issue9-phase4/PLAN.md` の §3.2 と §3.3、`src/beme/reranker.py`、`src/beme/converter.py`（`convert` の rerank 分岐 `:176-192`）、`scripts/diagnose_cases.py`（d1 JSONL の読み方・atomic JSON・SHA-256 の型）、`scripts/oracle_recall.py` を読むこと。

## 依存

Task 1（Top-1 定義の統一）。Task 3（集計モジュール）と並行してよいが、`combine` の集計は Task 3 の関数を使う。**先に着手した側が仮実装を置き、後から片方に寄せる**のではなく、**関数シグネチャを PLAN §3.5 の表で固定してから両方が実装する**こと。

## 対象ファイル

- 新規: `scripts/rerank_eval.py`
- 新規: `tests/test_rerank_eval.py`

`src/beme/` 配下は変更しない。**特に `CrossEncoderReranker` を改造しないこと。** 純粋な relevance は既存の重みパラメータ（`viterbi_weight=0.0` / `cross_encoder_weight=1.0`）で取れる。

## 変更内容

### 1. `score` サブコマンド（重い・1 回だけ）

```text
scripts/rerank_eval.py score
  --data            d1 split JSONL（必須）
  --dictionary      既定 data/dictionary_v2.tsv
  --weights         局所モデルの重み（必須）
  --reranker        {mock, cross-encoder, llm}（既定 cross-encoder）
  --beam-width      既定 64
  --cross-encoder-model / --cross-encoder-cache-dir / --cross-encoder-local-files-only
  --model-path      llm 用
  --torch-threads   torch.set_num_threads に渡す（決定性と並列制御のため）
  --output          JSONL 出力先（必須）
```

処理:

1. 各例で `Converter.convert(reading, k=beam_width, allow_literals=True)`（**リランカを注入しない converter**）で候補集合を得る。
2. `ConversionContext(preceding=left_context, following=right_context)` を作る。
3. リランカを `viterbi_weight=0.0` / 相手側の重み `1.0` で構築し、`rerank(candidates, context)` を呼ぶ。返る `score` を符号反転して `relevance` にする。
   - `MockContextReranker` は `score = cost − bonus` なので `relevance = cost − score`。
   - `LLMLogitsReranker` は `viterbi_weight=0` / `llm_weight=1` で `score = −mean_log_likelihood`。
   - **3 種すべてが同じ JSONL 形式に落ちること。**
4. `context.is_empty` の例（d1 test で 18 件）は**リランカを呼ばず** `relevance: null` を記録する。**0 を入れてはならない。**

出力 JSONL（1 例 1 行・`ensure_ascii=False`・キーソート）:

```json
{"id": "...", "reading": "...", "gold": "...",
 "contrast_pair_id": null, "span_tokens": 1, "reading_length": 5,
 "context_empty": false, "left_context_length": 44, "right_context_length": 6,
 "candidates": [{"surface": "出発", "cost": 2.25, "relevance": 7.709}, ...],
 "elapsed_seconds": 3.02}
```

先頭に 1 行のヘッダレコード（またはサイドカー JSON）で provenance を残す: データの path/件数/SHA-256、辞書 SHA-256、重みの path/feature_spec/SHA-256、リランカ種別とモデル名、**取得したモデルのリビジョン**、`beam_width`、`torch_threads`、実行機のホスト名と CPU コア数。

### 2. `combine` サブコマンド（軽い・何度でも）

```text
scripts/rerank_eval.py combine
  --scores          score の出力 JSONL（1 個以上・複数指定でラベル付き比較）
  --viterbi-weight  カンマ区切り。既定 0,0.25,0.5,1,2,3.94,8
  --rerank-k        カンマ区切り。既定 5,10,20,64
  --gate            任意。confidence 閾値のカンマ区切り
  --output          JSON 出力先（必須）
  --json            標準出力にも出す
```

各 `(vw, rerank_k)` について、`Converter.convert` の順位付け規則を**厳密に再現**する:

```text
head = candidates[:rerank_k]
head = sorted(head, key=lambda c: (vw*c.cost - ce_weight*c.relevance, c.cost, c.surface))
order = head + candidates[rerank_k:]
top1 = order[0].surface
```

このキーは `reranker.py:301`（`sorted(scored, key=(score, cost, surface))`）と `converter.py:176-192`（head を rerank し tail をそのまま連結）の合成である。

`context_empty` の例は**リランクしない**（`order = candidates`）。件数を出力に残す。

### 3. `combine` が出す指標（PLAN §3.3）

- **primary**: contrast pair 層（`contrast_pair_id is not None`）の Top-1。
- **secondary**: 全体の Top-1。
- **pair 単位**: contrast pair を `contrast_pair_id` で畳み、両方正解 / 片方 / 両方誤りの対数。
- **gold_rank 分布**: リランク前後。`{1: n, 2: n, ..., ">20": n, "absent": n}` の形。
- **recall 不変性のアサーション**: `surface_recall@k` が k ≥ `min(rerank_k, 候補数)` で**リランク前後で完全一致**すること。**破れたら例外にする**（候補集合の取り違えの検出）。
- **伸びしろ**: `surface_recall@rerank_k − surface_recall@1`（リランク前）。
- **層別**: contrast pair / 読み長（1-3 / 4-6 / 7-10 / 11-16）/ `span_tokens`。n < 30 は `low_support: true`。
- **失ったケース（b）と得たケース（c）の一覧**: id 昇順・全件（無作為抽出しない）。各件に reading / gold / before top1 / after top1 / gold_rank before・after を含める。
- **レイテンシ**: 候補あたり / 例あたり。`score` の `elapsed_seconds` から算出。**実行機のホスト名を必ず併記する。**
- **McNemar**: Task 3 の関数を使う。同一 id ペアリング。

### 4. 決定性

- JSON 出力は UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行。JSONL は 1 行 1 レコード・キーソート。
- atomic replace（`scripts/train_scorer.py:110-127` と同じ方式）。
- wall clock・絶対パス・ランダム temp 名を出力に入れない（`elapsed_seconds` は測定値なので例外。ただし `combine` の出力は `score` の値を引き写すだけで、`combine` 自身の時刻を入れない）。
- `combine` は同じ入力・同じ引数の 2 回実行で **byte-identical**。
- `score` は CE 推論のスレッド数で数値が揺れうる。`--torch-threads` を固定した上で 2 回実行し、**byte-identical なら「決定的」と報告、揺れるなら丸めずに揺れ幅を報告する**。

## 必須テスト

1. **`combine` の再現ソートが実 `Converter.convert` と一致する**（**最重要**）。fake なリランカ（決め打ちの relevance を返す test double）を `Converter` に注入して `convert` を呼び、同じ relevance を `combine` に食わせて Top-1 と全順位が完全一致することを、`rerank_k` が `None` / `< 候補数` / `>= 候補数` の 3 通りで固定する。
2. `context_empty` の例でリランカが呼ばれず `relevance: null` になり、`combine` が候補順を変えないこと。
3. recall 不変性のアサーションが、意図的に壊した入力で発火する。
4. `MockContextReranker` で `relevance = cost − score` が keyword bonus と一致する。
5. gold が候補集合に無い例で `gold_rank = "absent"` になり、Top-1 集計の分母に含まれること。
6. contrast pair の pair 単位集計が、片側しか無い対を正しく扱う（**扱い方を決めてテストで固定する**）。
7. `combine` の 2 回実行が byte-identical。
8. `score` が 3 種のリランカすべてで同じスキーマの JSONL を出す（LLM と CE は test double でよい。実モデルをテストで要求しないこと）。
9. `--viterbi-weight 0 --rerank-k 64` が「リランカ単独（R 条件）」に、リランカ無しが「局所単独（L 条件）」に一致すること。

## 完了条件

- `score` と `combine` が分離され、`combine` が推論を一切呼ばない。
- `combine` の順位付けが実 `Converter.convert` と一致することがテストで固定されている。
- 3 種のリランカが同じ JSONL 形式に落ちる。
- miniPC 上で、リポジトリ内の小さな合成 JSONL に対して `score` → `combine` が通る。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_rerank_eval.py
uv run python scripts/rerank_eval.py score --help
uv run python scripts/rerank_eval.py combine --help
uv run pytest -q

# 合成 fixture での疎通（mock リランカなら外部モデル不要）
uv run python scripts/rerank_eval.py score \
  --data tests/fixtures/<合成 split>.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json \
  --reranker mock --output /tmp/scores_mock.jsonl

uv run python scripts/rerank_eval.py combine \
  --scores /tmp/scores_mock.jsonl \
  --output /tmp/combine_mock.json --json

# 決定性
uv run python scripts/rerank_eval.py combine --scores /tmp/scores_mock.jsonl --output /tmp/c1.json
uv run python scripts/rerank_eval.py combine --scores /tmp/scores_mock.jsonl --output /tmp/c2.json
cmp /tmp/c1.json /tmp/c2.json && echo "byte-identical"

# 実 cross-encoder の疎通（miniPC のみ。~/.cache/beme/cross-encoder にキャッシュ済み）
# 数件の合成 fixture で 1 度だけ。d1 全件は Task 5 で GPU 側が回す
uv run python scripts/rerank_eval.py score \
  --data tests/fixtures/<合成 split>.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json \
  --reranker cross-encoder --cross-encoder-local-files-only \
  --torch-threads 4 --output /tmp/scores_ce.jsonl
```

## 報告に含めること

- 上記コマンドの**実出力**。
- テスト 1（実 converter との一致）が、わざと壊した `combine` で FAIL することの確認。
- `score` を実 cross-encoder で 2 回回したときの byte-identical 判定（同じなら「決定的」、違うなら揺れ幅）。
- 実 cross-encoder の 1 候補あたりレイテンシ実測（参考値: miniPC・N100 4 core で **63–72 ms/候補**）。

## 共通ルール・制約

- `git commit` / `git push` 禁止。
- `src/beme/` 配下を変更しない。特に `CrossEncoderReranker` を改造しない。必要性が判明したら勝手に広げず報告する。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。テスト用 fixture は「d1 のスキーマに合う小さな合成データ」であることを明示し、実データを名乗らせない。
- テストで実モデル（cross-encoder / LLM gguf）のロードを要求しない。test double を使う。
- 2,000 件の本番実行はこのタスクでは行わない（Task 5）。
