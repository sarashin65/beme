# Task 5: 測定実行（3 条件 × d1 dev/test・GPU マシン）

## 目的

Task 2 のハーネスで、**局所モデル単独 / リランカ単独 / 併用**の 3 条件を d1 で実測する。issue #9 Phase 4 の 3 つのチェックリスト項目のうち 2 つ（「K-best をリランカへ渡す」「3 条件を比較」）がここで完結し、残り 1 つ（「oracle recall と Top-1 改善を分けて評価」）の数値もここで出る。

**このタスクの成否は、実行前に選択規則を固定して守れるかで決まる。** Phase 3 の非目標「同じ 200 件でパラメータ選択と最終評価を行うこと」を再演しないこと。

最初に `docs/missions/issue9-phase4/PLAN.md` の §3.3・§3.4・§4・§5・§7 を読むこと。

## 依存

Task 1（Top-1 統一）・Task 2（ハーネス）・Task 3（集計モジュール）・Task 4（リーク監査）がすべて完了していること。

## 実行環境

GPU マシン `beme-gpu`（24 core / 31 GB）。作業ディレクトリは Task 4 で用意した `~/beme-p4`（本ブランチの git クローン + `data/d1` 配置済み）。Task 1〜3 の変更を取り込んでから実行すること。

**cross-encoder モデルは GPU マシンにキャッシュされていない。** ネットワークは到達確認済みなので、`--cross-encoder-local-files-only` を付けずに 1 回だけ取得する。**取得後、モデルのリビジョンとファイルの SHA-256 を artifact に記録すること。**

局所モデルの重み: `~/beme-p3/artifacts/issue9-phase3/A0/seed{9,17,33}/weights.json`。seed 9 の SHA-256 は `70a6ffed5405bbc4747e83eb36bc0a7213800263ce1db5e2c47d8dd438145fee`。

## 実行前に固定すること（実行後に変えない）

次をこのタスクの最初のコミット可能な成果物として `artifacts/issue9-phase4/selection_rule.json` に書き出し、**実行前にユーザーへ報告する**。

```json
{
  "selection_split": "d1 dev (1000 examples)",
  "selection_primary": "overall top1 accuracy",
  "selection_secondary": "contrast pair top1 accuracy (44 examples)",
  "tie_break": ["overall_top1 desc", "contrast_top1 desc", "rerank_k asc", "viterbi_weight desc"],
  "grid": {"viterbi_weight": [0, 0.25, 0.5, 1, 2, 3.94, 8], "rerank_k": [5, 10, 20, 64]},
  "test_evaluations": 1,
  "seeds_for_stability": [9, 17, 33],
  "seed_selection_policy": "select on seed 9 dev only; apply the same (vw, rerank_k) to seeds 17 and 33 without re-selection"
}
```

## 実行マトリクス

### `score` フェーズ（CE 推論・重い）

| # | split | 重み | リランカ | 見積り |
|---|---|---|---|---:|
| S1 | `data/d1/dev.jsonl`（1,000） | A0 seed 9 | cross-encoder | 10–15 分 |
| S2 | `data/d1/test.jsonl`（2,000） | A0 seed 9 | cross-encoder | 20–30 分 |
| S3 | `data/d1/test.jsonl` | A0 seed 17 | cross-encoder | 20–30 分 |
| S4 | `data/d1/test.jsonl` | A0 seed 33 | cross-encoder | 20–30 分 |
| S5 | `data/d1/test.jsonl` | A0 seed 9 | mock（対照） | 1 分未満 |

S2/S3/S4 は独立なので 3 プロセス並列でよい。**`CrossEncoderReranker` は内部 `_lock` で直列化されているので 1 プロセス内では並列化できない。** プロセスを分け、`--torch-threads 8` で合計 24 コアを超えないようにすること。

出力先は条件ごとに別ディレクトリ（`artifacts/issue9-phase4/{S1..S5}/`）。同一ディレクトリへ複数プロセスを向けないこと（atomic replace の temp が衝突する）。

### `combine` フェーズ（推論ゼロ）

| # | 入力 | 内容 |
|---|---|---|
| C1 | S1 | グリッド全面 7 × 4 = 28 点。**選択はここで 1 回だけ。選択規則に従って 1 点を決める** |
| C2 | S2 | 選択した 1 点で L / R / LR の 3 条件。全指標 |
| C3 | S3, S4 | C2 と同じ 1 点。seed 安定性 |
| C4 | S5 | mock 対照 |
| C5 | S2 | ゲート候補の recall / fire_rate 曲線（Task 6 の入力） |

**3 条件の定義**（PLAN §3.4）:

| 条件 | 定義 |
|---|---|
| **L: 局所モデル単独** | リランカを通さない。候補順は局所コスト順 |
| **R: リランカ単独** | 候補生成は L と同一。順位付けは relevance のみ（`viterbi_weight = 0`・`rerank_k = 64`） |
| **LR: 併用** | C1 で選択した `(viterbi_weight, rerank_k)` |

### 既存 suite の回帰確認

Task 1 の変更後に 1 回。**採否判断には使わない**（Mozc は context を持たずリランカが走らない、handmade は mock で既に Top-1 1.000、AJIMEE は文分割が混ざる）。

```bash
uv run python scripts/evaluate.py --suite all --reranker mock \
  --dictionary data/dictionary_v2.tsv --weights data/weights/decomposed_v1.json --json
# 期待: Handmade NDCG@5 0.992972 / Top-1 1.000000、AJIMEE CER 0.290080 / exact 0.070000、Mozc 88/498
```

## レイテンシの測定（miniPC で別途）

**レイテンシは実験の性質ではなく実行機の性質である。** GPU マシンの `elapsed_seconds` は「スコアリングにかかった時間」として記録するが、**IME としてのレイテンシは miniPC（Intel N100・4 core）で測る**。

miniPC で d1 test の先頭 50 件程度に `score` を回し、候補あたり / 変換あたりのレイテンシと、`rerank_k ∈ {5, 10, 20, 64}` による削減を出す。参考実測値: **63–72 ms/候補・約 3.0 秒/変換（ユニーク候補 47.9）**。

**レポートの全レイテンシ数値に実行機名を併記すること。** GPU マシンの値と miniPC の値を混ぜないこと。

## 完了条件

- `selection_rule.json` が実行前に固定され、報告されている。
- S1〜S5 の `score` が完走し、artifact が保存されている。
- C1 のグリッド全面（28 点）が出ており、**選択点が高原の上にあるかスパイクか**が判定されている。
- C2 が 3 条件（L / R / LR）の primary（contrast 128）・secondary（全体 2,000）・pair 単位（64 対）・gold_rank 分布・層別・**失ったケース（b）の全件一覧**を出している。
- C3 で 3 seed の向きが一致するかが出ている。
- **リランク前後で `surface_recall@k`（k ≥ min(rerank_k, 候補数)）が不変**であることが実測アサーションで確認されている。
- PLAN §5 の 3 分岐のどれに該当するかが判定されている。

## 検証コマンド

```bash
# GPU マシン
ssh beme-gpu
cd ~/beme-p4 && git fetch && git checkout feat/issue9-phase4 && git pull
sha256sum data/d1/*.jsonl artifacts/../../beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json

# S2 の例
uv run python scripts/rerank_eval.py score \
  --data data/d1/test.jsonl \
  --dictionary data/dictionary_v2.tsv \
  --weights ~/beme-p3/artifacts/issue9-phase3/A0/seed9/weights.json \
  --reranker cross-encoder --torch-threads 8 \
  --output artifacts/issue9-phase4/S2/scores.jsonl

# C1（選択）
uv run python scripts/rerank_eval.py combine \
  --scores artifacts/issue9-phase4/S1/scores.jsonl \
  --viterbi-weight 0,0.25,0.5,1,2,3.94,8 --rerank-k 5,10,20,64 \
  --output artifacts/issue9-phase4/C1/grid.json --json

# C2（選択後 1 回）
uv run python scripts/rerank_eval.py combine \
  --scores artifacts/issue9-phase4/S2/scores.jsonl \
  --viterbi-weight <選択値> --rerank-k <選択値> \
  --output artifacts/issue9-phase4/C2/test.json --json

# miniPC 側でレイテンシ
uv run python scripts/rerank_eval.py score \
  --data <test の先頭 50 件> --dictionary data/dictionary_v2.tsv \
  --weights <A0 seed9 weights> --reranker cross-encoder \
  --cross-encoder-local-files-only --torch-threads 4 \
  --output /tmp/latency_minipc.jsonl
```

## 報告に含めること

- `selection_rule.json` の全文（**実行前に**別途報告済みであること）。
- 全コマンドの逐語と**実出力**。
- 入力の SHA-256（d1 3 ファイル・辞書・重み 3 本）と、取得した cross-encoder のモデル名・リビジョン。
- C1 のグリッド全面の表（28 点）。
- C2 の 3 条件比較の表（primary / secondary / pair 単位）と McNemar の `b` / `c` / `p`。
- C3 の seed 3 本の向き。
- 失ったケース（b）の全件一覧。
- `surface_recall@k` 不変性アサーションの結果。
- レイテンシ（**miniPC 実測・実行機名併記**）。
- PLAN §5 のどの分岐に該当するかの判定と、その根拠。

## 共通ルール・制約

- `git commit` / `git push` 禁止。
- **選択は d1 dev で 1 回。d1 test は選択後に 1 回だけ評価する。** グリッド最良点を test で報告しない。
- **train の contrast 872 件を選択に使わない**（局所モデルが train で学習済みなので候補コストが楽観側に偏り、`viterbi_weight` の最適値が系統的にずれる）。使わない理由をレポート材料として残すこと。
- 既存 suite（AJIMEE / Mozc / handmade）を採否判断に使わない。回帰検出の補助のみ。
- 局所モデルを再学習しない。Phase 3 の A0 の重みをそのまま使う。
- 欠損を推測で補わない。プロセスが落ちた条件は `failed` として summary に残す（Phase 3 と同じ規約）。
- d1 JSONL を repo にコミットしない。
