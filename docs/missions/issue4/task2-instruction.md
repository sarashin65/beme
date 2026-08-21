# Task 2: 頻度集計とコスト減算スクリプトの実装（issue #4 フェーズ1）

## 指示

`docs/missions/issue4/PLAN.md` の Task 2 を実装してください。Task 1（`scripts/fetch_corpus.py`・exit 134 修正済み）は完了済みです。

- `scripts/apply_cost_learning.py` を作成し、`fugashi` 等で `data/corpus.txt` を並列解析して頻度集計を行う
- 上位 5,000 語のコストを減算するロジックを書き、既存の `data/dictionary.tsv` を更新する
- 頻度集計の粒度: `(reading, surface, pos)` タプル（`src/beme/ipadic.py` の `katakana_to_hiragana` / `map_ipadic_pos` を利用）
- コスト減算は対数減衰（例: ΔC = α × log10(f)）または定数減算から開始し、`data/dictionary.tsv` の既存エントリに適用
- BeME のコスト下限を下回らないようクリッピングすること

## 完了条件

- `scripts/apply_cost_learning.py` が動作し、`data/dictionary.tsv` の高頻度語のコストが更新される
- 更新後の辞書で `scripts/evaluate.py --suite handmade` が通る（既存の手作りケースを壊さない）
- 更新前後の diff を確認できる（どれだけの語が・どのように変わったか）

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- fugashi が必要なら `pyproject.toml` に依存を追加してよい
- 実装後は自己確認（スモーク実行・diff 確認）して報告
