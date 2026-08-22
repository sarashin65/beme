# Task 2: 学習 JSONL・split・manifest 契約

## 目的

train/dev/test を厳格に読み、データ欠損や split leakage を黙って受け入れず、入力 bytes と分割を再現可能な manifest にする。

最初に `README.md`、`docs/missions/issue9-phase2/PLAN.md` §1.3・§2.2、`docs/missions/p5/PLAN.md` §4.1–4.4 を読むこと。現 branch には `data/d1/*.jsonl` が存在しない点を前提にする。

## 対象ファイル

- 新規: `src/beme/training_data.py`
- 新規: `tests/test_training_data.py`
- 新規 fixture は `tests/data/phase2/` 配下のみ
- 必要な公開 export に限り: `src/beme/__init__.py`

実データ、辞書、既存評価データ、`.gitignore`、学習器は変更しない。

## 変更内容

1. frozen/slots の `TrainingExample` と strict JSONL loader を実装する。
2. 非空 `reading` と正解表記を必須にする。canonical key は `target_surface`。#5 互換として `positive` も受け、両方ある場合は同値必須。
3. `id`, `source`, `article_id`, `left_context`, `right_context`, `candidates`, `viterbi_costs`, `frequency_bucket` を許可する。型を検証し、未知 key は schema drift として行番号付きエラーにする。
4. blank line、非 UTF-8、duplicate JSON key、非 object、重複 id を拒否する。
5. 3 split loader は split 間の `article_id` 交差を拒否する。article ID が不足する場合は検証済みと偽らない。
6. split ごとの相対/指定 path、SHA-256、件数、source 集計、`article_disjoint_verified` を持つ決定的 manifest を返す。絶対 cwd や時刻は入れない。
7. context は保持するが Phase 2 local scorer の入力ではないことを docstring に書く。

## 必須テスト

- `target_surface` と `positive` の正常系、両者不一致。
- UTF-8、blank、duplicate key/id、不正型、未知 key の各エラーに行番号がある。
- article ID 交差の検出と、ID 不足時の `article_disjoint_verified=false`。
- 同じ bytes の manifest が byte-equivalent な dict になる。
- 10k を作らず、小さな fixture で件数と SHA-256 を固定する。
- `data/d1` 不在を synthetic data へ黙って置換しない。

## 完了条件

- #5 形式と canonical pair 形式が同じ `TrainingExample` になる。
- split の provenance と bytes が report へ渡せる。
- 欠損 path は明確な `OSError`/domain error で失敗する。

## 検証コマンド

```bash
uv run pytest -q tests/test_training_data.py
uv run pytest -q tests/test_evaluation.py
```

## 共通ルール・制約

- `git commit` / `git push` 禁止。変更は作業ツリーに残す。
- 対象外ファイルを変更しない。特に存在しない `data/d1` を生成・取得・コミットしない。
- README のライセンス/provenance 方針を守る。
- テストを実装と同時に追加し、検証結果を報告する。
- formatter の一括適用や既存 JSONL の書換えをしない。

