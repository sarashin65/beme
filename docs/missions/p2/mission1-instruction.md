# Mission 1: LatticePath の永続化と片方向リスト化

## 指示

`docs/missions/p2/PLAN.md` の **Mission 1** を実装してください。

- `src/beme/lattice.py` を修正し、`LatticePath` を片方向リスト（`token`, `parent`）構造にする
- プロパティ `tokens`, `surface`, `reading` は、要求された時のみリストを辿って構築する（遅延評価）
- エッジ追加ごとの tuple 完全コピーをやめ、DP 遷移を O(1) にする

## 完了条件

- 既存のテスト・コマンドが現状通り動作する（挙動不変）
- `uv run pytest -q` が全件パスする

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（テスト実行）して報告
