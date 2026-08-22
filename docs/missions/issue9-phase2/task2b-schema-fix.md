# Task 2 修正指示: d1 実データスキーマ対応（sol 判断）

sol（司令塔）の設計判断に基づく修正指示です。**対象ファイルは src/beme/training_data.py と tests/test_training_data.py のみ**。それ以外は変更しない。git commit / git push 禁止。

## 背景

Task 6 の実データ学習が `data/d1/train.jsonl: line 1: unknown field: 'article_sha256'` で停止。Task 2 で実装した厳格 loader（未知 key は schema drift としてエラー）が意図どおり発動した。実データ（#5 作成 d1）の全フィールドは以下 16 個:

article_id / article_sha256 / candidates / context_truncated / contrast_pair_id / frequency_bucket / gold_index / id / left_context / positive / reading / right_context / source / span_tokens / split / viterbi_costs

現行許可リストには id / source / article_id / left_context / right_context / candidates / viterbi_costs / frequency_bucket（+ positive alias・target_surface canonical）が既にある。**追加で許容が必要な 6 フィールド**: article_sha256 / context_truncated / contrast_pair_id / gold_index / span_tokens / split

## 修正内容

### 1. 許可フィールド追加

_ALLOWED_KEYS へ以下を追加する:
- article_sha256
- context_truncated
- contrast_pair_id
- gold_index
- span_tokens
- split

### 2. TrainingExample へ default 付きフィールド追加

既存 fixture との後方互換を維持する:
- article_sha256: str | None = None
- context_truncated: bool | None = None
- contrast_pair_id: str | None = None
- gold_index: int | None = None
- span_tokens: tuple = ()
- split: str | None = None

これらは provenance/検証用であり、Phase 2 scorer の特徴には使わない。

### 3. 行単位 validation

- article_sha256: 省略可・指定時は 64 文字の 16 進文字列
- context_truncated: 省略可・指定時は厳密な bool（int の 0/1 は拒否）
- contrast_pair_id: 省略/null 可・非 null なら非空文字列
- gold_index: 省略可・bool ではない非負整数・candidates の範囲内・candidates[gold_index] == canonical target_surface・candidates なしで指定された場合はエラー
- span_tokens: 省略可・指定時は配列・tuple へ変換して保持・内容は Phase 2 学習では解釈しない
- split: 省略可・指定時は train/dev/test のいずれか

validation error には従来どおり path と line number を含める。

### 4. split 横断 validation

load_splits で以下を検証する:
- 各行に split がある場合、読み込み先の train/dev/test と一致すること
- 同じ article_id に異なる article_sha256 が対応しないこと
- 同じ contrast_pair_id が複数 split に現れないこと

split フィールドが省略された既存 synthetic fixture は引き続き許可する。ただし、値が存在して不一致なら必ずエラーにする。

### 5. manifest

既存項目を維持し、最低限以下を追加する。

split 別:
- context_truncated_count
- contrast_pair_count
- article_sha256_present_count
- span_tokens_present_count
- split_field_verified

トップレベル:
- article_sha256_consistent
- contrast_pairs_disjoint
- split_field_verified

ID や SHA の全一覧は出力せず、件数と検証結果だけを記録する。JSON の決定性を維持する。

## 必須テスト

1. 実データと同じ全 16 フィールドを持つ行をロードできる
2. positive が target_surface へ canonical 化される
3. 未追加の未知フィールドは引き続き拒否される
4. article_sha256 の形式不正
5. 同一 article_id で article_sha256 不一致
6. context_truncated の非 bool
7. contrast_pair_id の split 跨ぎ
8. gold_index の負値・範囲外・positive との不一致
9. span_tokens が配列でない場合
10. split 値不正・読み込み先 split との不一致
11. 既存 synthetic fixture が変更なしで通る
12. manifest が 2 回生成で同一

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
