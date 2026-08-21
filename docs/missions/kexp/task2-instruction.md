# Task 2: オフライン実験の実行と結果保存（k 拡大ゲート実験）

## 指示

`docs/p2-kexp-plan.md` の Task 2 を実施してください。Task 1（`--k` フラグ対応）は完了済みです。

- 対象: AJIMEE 200件（`data/ajimee_bench_jwtd_v2_v1.json`）
- 条件: k = 5, 10, 20, 64 × `--reranker cross-encoder`（既定モデル japanese-reranker-base-v2）
- 各条件の評価結果（推論結果 + メトリクス）を JSON 形式で worktree 内に保存する（例: `results_k10.json` 等）
- レイテンシ（リランカ推論時間・全体処理時間）も記録する

## 完了条件

- k=5, 10, 20, 64 の全条件で AJIMEE 200件の評価が完了している
- 各条件の結果 JSON が worktree 内に保存されている
- 実行コマンドと結果の要約（CER・Top-1 完全一致率・レイテンシ）を報告に含める

## 制約

- git commit / push はしない
- production コードは変更しない（Task 1 の実装を壊さない）
- cross-encoder モデルは `~/.cache/beme/cross-encoder/` にある（repo 外）
