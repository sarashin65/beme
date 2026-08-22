# Task 2 補完: 辞書スキーマ拡張の回帰テスト追加（issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 2 の完了条件をテストで保護してください。実装（`src/beme/dictionary.py` の Task 2 差分）は既に済んでいます。

## 追加するテスト

`tests/test_dictionary_lattice.py`（または必要なら新規テストファイル）に以下を追加:

1. **任意列付き TSV が読め、欠損列は既定値になる**: 4 列 + 任意列（`pos_detail` / `conjugation` / `left_id` / `right_id` / `source` / `raw_cost`）を含む TSV を読み込み、各フィールドが正しくパースされること。一部の列が無い場合は既定値（`pos_detail=""` / `conjugation=""` / `left_id=-1` / `right_id=-1` / `source=""` / `raw_cost=None`）になること
2. **任意列は順不同で書ける**: ヘッダで任意列の順序を変えても正しく読めること
3. **未知列名が行番号付きでエラー**: 未知の列名を含む TSV が `path: line N:` 形式の ValueError で失敗すること
4. **不正型が行番号付きでエラー**: `left_id` に負値・非整数、`source` に enum 外（`ipadic` / `overlay` / `learned` / `manual` 以外）が `path: line N:` 形式の ValueError で失敗すること
5. **4 列形式の後方互換**: 既存の 4 列 TSV が従来どおり読める（既存テストでカバー済みのはずだが、明示ケースがあれば追加）

## 検証

```bash
uv run pytest -q tests/test_dictionary_lattice.py tests/test_scorer_golden.py
uv run pytest -q   # 全件 pass を確認
```

## 制約

- git commit / push しない
- `src/beme/` の production code は変更しない（テスト追加のみ）
- 自己確認した出力（pytest 結果）を貼って報告すること
