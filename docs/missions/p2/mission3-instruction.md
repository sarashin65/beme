# Mission 3: Eppstein (A*) による Exact K-best の実装

## 指示

`docs/missions/p2/PLAN.md` の **Mission 3** を実装してください。Mission 1（LatticePath 永続化）・Mission 2（backward_viterbi 前計算）は完了済みです。

- `k_best_viterbi`（または同等の経路探索関数）を、優先度付きキュー（`heapq`）を用いた **A* 探索**に置き換える
- Mission 2 で得た `backward_viterbi` の最小コストをヒューリスティックとして使い、始点から K 個のパスが抽出されるまで **Exact な順序**で探索を行う
- `beam_width` による打ち切りを廃止する（または Exact を保つ形で無害化する）
- 抽出されたパスの順序はコスト昇順（1-best が最小コスト）を保証すること

## 完了条件

- K-best の順序が Exact（コスト昇順・打ち切りなし）であることのテストを追加
- 既存のテストが全件パスする（`uv run pytest -q`）
- 既存 CLI（`beme` 変換コマンド）の挙動が維持される（出力・コストが従来と同等 or 改善）

## 制約

- git commit / push はしない（作業ツリーに変更を残す）
- 対象外ファイルを変更しない
- 実装後は自己確認（テスト実行・CLI スモーク）して報告
