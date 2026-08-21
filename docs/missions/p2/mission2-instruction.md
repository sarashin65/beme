# Mission 2: Viterbi 前計算 (Backward DP) の実装

## 指示

`docs/missions/p2/PLAN.md` の **Mission 2** を実装してください。Mission 1（LatticePath の片方向リスト化）は完了済みです。

- `src/beme/lattice.py` に、終端ノードから始点ノードに向かって「終端までの最小コスト」を計算する Backward DP 処理を追加する
- この値は Mission 3 で実装する A* 探索のヒューリスティック関数として利用する
- 接続コスト（pos_transition_cost）・生起コストを正しく反映した「終端までの最小コスト」を各ノードについて計算する

## 完了条件

- Backward DP が正しく実装されている（テストで検証: 各ノードの最小残余コストが Forward の総コストと整合する等）
- 既存のテストが全件パスする（`uv run pytest -q`）
- 必要なテストを追加する

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（テスト実行）して報告
