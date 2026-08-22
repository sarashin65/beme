# Task 2b 修正指示 2: span_tokens を int に修正 + 全フィールド一括監査（sol 判断）

**対象ファイルは src/beme/training_data.py と tests/test_training_data.py のみ**。それ以外は変更しない。git commit / push 禁止。

## 背景

実データの d1 JSONL を load_splits で読むと `span_tokens must be an array` で失敗。実データの span_tokens は**全件 int（トークン数・1 以上）**であることが 500 件サンプルで確認済み。sol は「span_tokens だけを場当たり的に直す前に**全フィールドを一括監査**せよ」と指示している。次の未知の型差による逐次停止を避けるため。

## 修正内容

### 1. 全フィールド一括監査（先にやる）

GPU マシン `~/beme-p2` の `data/d1/train.jsonl`（先頭 500 行でよい）を読み、各フィールドの型・値の実態を確認する。確認項目:

- article_sha256: 64 桁小文字 hex・同一 article_id 内で一貫
- context_truncated: type(value) is bool
- contrast_pair_id: null または非空 str・同じ ID の split 跨ぎがない
- gold_index: type(value) is int・0 以上・候補範囲内・candidates[gold_index] == canonical target_surface
- split: train/dev/test 限定
- candidates / viterbi_costs: 同じ長さ・cost は有限の int/float で bool 不可
- span_tokens: int・1 以上（下記修正後）

監査結果が上記期待条件を満たす場合、既存 Task 2b 契約を維持する。**異なる場合、実データに合わせて黙って union を増やさず、結果を報告して判断を仰ぐ**。

### 2. span_tokens の修正

TrainingExample の span_tokens を次に変更する:

```python
span_tokens: int | None = None
```

validation:
- 省略可（既存 synthetic fixture との後方互換用）
- 指定時は type(value) is int
- bool は拒否
- 1 以上
- 配列、tuple、float、文字列は拒否
- Phase 2 scorer の特徴には使わない

以前追加した配列→tuple 変換処理は削除する。配列形式を後方互換として残さない。

### 3. manifest

既存の span_tokens_present_count を維持し、split 別に追加する:
- span_tokens_min
- span_tokens_max

値が全件欠落する旧 fixture では両方 null。実データでは整数値を記録する。

### 4. 必須テスト修正

- span_tokens=1 を正常にロードできる
- span_tokens=0、-1、True、1.0、"1"、[] を拒否する
- span_tokens 省略の既存 fixture は引き続き通る
- manifest の min/max を検証する
- 全 16 フィールドを持つ実データ相当行をロードできる
- gold_index と positive の整合性を検証する
- 未知フィールド拒否を維持する

以前作成した「span_tokens 配列を受ける」テストは、期待値を int へ修正する。配列対応テストとして残さない。

## 検証コマンド

```bash
uv run pytest -q tests/test_training_data.py
uv run pytest -q tests/test_train_scorer_cli.py tests/test_phase2_integration.py
uv run pytest -q
```

## 制約

- d1 JSONL を変換・上書きしない
- 未知フィールドを全面許可しない
- metadata を学習特徴へ追加しない
- validation を warning へ格下げしない
- 対象外ファイルを変更しない
- git commit / git push 禁止

この修正後、Task 6 は元の data/d1/*.jsonl をそのまま入力して再開する。
