# Task 4: oracle recall@k 測定 CLI

## 目的

issue #9 Phase 3 チェックリストの「学習後の oracle recall@k を測定」を実装する。issue の Phase 3 注意点 3 が要求する「oracle recall@k は条件付きランキング精度と分けて測る」を、**3 つの別々の量として定義し、混同できない形で出力する**。

最初に `docs/missions/issue9-phase3/PLAN.md` の §3.4、`src/beme/oracle.py`、`src/beme/lattice.py`（`k_best_viterbi`）、`src/beme/converter.py`（`convert` の重複排除）、`src/beme/training.py`（`evaluate_split`）、`src/beme/training_data.py`、`scripts/train_scorer.py` を読むこと。

## 依存

なし。Task 1〜3 と並行して進めてよい。

## 測定する 3 つの量（混同しないこと）

| 指標 | 定義 | 依存 |
|---|---|---|
| **lattice coverage** | `constrained_best_path(lattice, gold, scorer) is not None` | k に依存しない |
| **path recall@k** | `k_best_viterbi(lattice, beam_width=k, scorer)` が返す経路のいずれかの `surface` が gold と一致 | k と重みに依存 |
| **surface recall@k** | `Converter.convert(reading, k=k)` の候補 surface に gold が含まれる | k・重み・`beam_width` に依存 |

**`path recall@k` と `surface recall@k` は別物である。** `Converter.convert`（`converter.py:164-170`）は `beam_width` 本の経路を取ってから surface で重複排除し、その後 `[:k]` する。同じ surface を作る経路が複数あるので、`beam_width=64` でもユニーク surface は 64 未満になる。**Phase 4 のリランカが実際に受け取るのは 3 つ目である。** したがって出力には k と `beam_width` を必ず両方記録すること。

## 対象ファイル

- 新規: `scripts/oracle_recall.py`
- 新規: `tests/test_oracle_recall.py`

`src/beme/` 配下は変更しない。既存の公開 API（`constrained_best_path`・`k_best_viterbi`・`Converter.convert`・`load_splits`）だけで実装できる。できないと判断したら、勝手に広げず報告すること。

## 変更内容

### 1. CLI

```text
--data            gold を持つ JSONL（d1 の split ファイル）。--data か --suite のどちらか必須
--suite           {mozc, handmade}（gold 表記が取れる既存 suite）
--dictionary      既定 data/dictionary_v2.tsv
--weights         必須。baseline / 学習済みを明示的に指定させる
--k               カンマ区切り。既定 1,5,10,20,64
--beam-width      既定 64
--output          JSON 出力先（必須）
--json            標準出力にも JSON を出す
```

- `--data` は `beme.training_data.load_splits` の契約に合わせる（単一ファイルを読む API が無ければ、既存 loader を単一 split で使う形にする。**loader を改造しない**）。
- **AJIMEE は対象外**。`evaluate_ajimee` は `convert_sentence` で文単位に分割して変換する構造であり、単一読みに対する recall と定義が合わない。`--suite ajimee` は明示的なエラーメッセージで拒否し、理由を文言に含めること。

### 2. 出力 JSON

条件（`weights` の spec id と SHA-256）、`dictionary` の SHA-256、データの件数と SHA-256、`beam_width`、各 k について 3 指標を含める。

```json
{
  "dataset": {"path": "...", "examples": 2000, "sha256": "..."},
  "dictionary_sha256": "...",
  "weights": {"path": "...", "feature_spec": "...", "sha256": "..."},
  "beam_width": 64,
  "lattice_coverage": {"covered": 2000, "examples": 2000, "rate": 1.0},
  "path_recall_at_k": {"1": 0.731, "5": ..., "10": ..., "20": ..., "64": ...},
  "surface_recall_at_k": {"1": ..., "5": ..., "10": ..., "20": ..., "64": ...},
  "unique_surface_count": {"mean": ..., "p50": ..., "p90": ..., "max": ...}
}
```

`unique_surface_count` は `beam_width` 本の経路が何本のユニーク surface に潰れるかの分布。Phase 4 が `rerank_k` を決めるのに必要なので必ず出す。

### 3. 決定性

- 出力 JSON は UTF-8・`ensure_ascii=False`・`sort_keys=True`・`indent=2`・末尾改行。
- 一時ファイルへ書いて atomic replace（`scripts/train_scorer.py:110-127` の `_atomic_json` と同じ方式）。
- wall clock・絶対パス・ランダム temp 名を出力に入れない。
- 同じ入力・同じ引数の 2 回実行で byte-identical。

### 4. 整合性チェック（必須）

次を実行時に検証し、破れたら例外にすること。バグの早期検出になる。

- `path_recall@k` は k について単調非減少。
- `surface_recall@k` は k について単調非減少。
- `path_recall@k <= lattice_coverage rate` が全 k で成立（ラティスに無い gold は K-best にも出ない）。
- `surface_recall@k <= path_recall@beam_width` が全 k で成立。
- `path_recall@1` が `1 - model_error_rate`（`beme.training.evaluate_split` の値）と一致する。

## 必須テスト

- 小さな合成ラティスで 3 指標を手計算値と照合。
- gold がラティスに存在しない例で `lattice coverage` が落ち、`path_recall@k` が全 k で 0。
- 同じ surface を作る経路が 2 本ある例で、`path recall` と `surface recall` の分母・挙動が異なることを固定。
- `unique_surface_count` が `beam_width` 以下。
- k の単調性と 4 つの整合性チェックが、意図的に壊した fake 入力で発火する。
- `--suite ajimee` が非 0 終了し、理由がメッセージに出る。
- 2 回実行が byte-identical。
- 別プロセスで学習済み重みを読んで動く。

## 完了条件

- 3 指標が定義どおり分離して出力される。
- k と `beam_width` の両方が出力に記録される。
- d1 test 2,000 件・Mozc 498 件・handmade 14 件で動く。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_oracle_recall.py
uv run python scripts/oracle_recall.py --help
uv run pytest -q

# handmade（repo 内データのみで動く）
uv run python scripts/oracle_recall.py --suite handmade \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json \
  --output /tmp/oracle_handmade.json --json

# Mozc
uv run python scripts/oracle_recall.py --suite mozc \
  --dictionary data/dictionary_v2.tsv \
  --weights data/weights/decomposed_v1.json \
  --output /tmp/oracle_mozc.json --json

# d1 test（GPU マシンでのみ実行可能。miniPC には data/d1/*.jsonl が無い）
# uv run python scripts/oracle_recall.py --data data/d1/test.jsonl \
#   --dictionary data/dictionary_v2.tsv \
#   --weights data/weights/decomposed_v1.json \
#   --output artifacts/issue9-phase3/oracle/baseline-test.json
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- `src/beme/` 配下を変更しない。必要性が判明したら勝手に広げず報告する。
- `data/d1/*.jsonl` を生成しない・推測で合成しない。miniPC には実データが無いので、実データ検証は GPU マシン（Task 8）で行う。テストはリポジトリ内の小さな合成 fixture で完結させること。
- AJIMEE を無理に対象へ含めない。
- ネットワーク・GPU・外部モデルを必要としない CLI/テストにする。
- テストを実装と同時に追加し、上記コマンドの実結果を報告する。
