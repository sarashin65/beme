# Mission: Phase 4.1 レポート AMEND（A8 ベース確定）+ ハーネス修正

## 背景
issue #17 Phase 4.1 で A7/A8 の seed 増比較（10 seed・事前登録）を実行し、
**A8（`decomposed_v1-no_script-surface_len`）が 10/10 seed で A7 を下回り、両側符号検定 p=0.001953 < .05 で新ベースに AMEND された**。

作業環境: `~/work/beme-4p1`（branch `feat/issue9-phase4.1`・origin/main @ aabd057 起点）
材料: `gh issue view 17`（事前登録ルール）・`docs/missions/issue9-phase4.1/phase4.1-report.md`（現行レポート・「A7 採用」と書かれている）

## 10 seed 結果（正本データ・この表をレポートに反映する）
A7 error: seed1=531 / 5=524 / 9=530 / 13=527 / 17=532 / 21=529 / 25=528 / 29=531 / 33=529 / 37=532（中央値 529.5）
A8 error: seed1=520 / 5=523 / 9=521 / 13=518 / 17=520 / 21=519 / 25=523 / 29=520 / 33=520 / 37=525（中央値 520.0）
A8 < A7 は 10/10 seed（Δ -1〜-12）・両側符号検定 p = 0.001953
元データ: GPU マシン `~/beme-p41/artifacts/issue9-phase4.1-seeds10/A7|A8/seed{1,5,9,13,17,21,25,29,33,37}/report.json` の `selected_model.test.end_to_end_errors`

## Task 1: ハーネス修正（scripts/run_ablations.py）
`summarize_only()`（143-145 行付近）が SEEDS 定数 (9,17,33) をハードコードしており、`--seeds` 引数を無視する。
`--summarize-only` 実行時に指定された `--seeds` を使うよう修正し、テストを追加する:
- `--summarize-only --seeds 1,5,9` が seed 1/5/9 だけを集計すること
- seed が欠損している条件は available 判定から外れる既存挙動を維持すること
- 既存テスト（tests/test_run_ablations.py）が全て通ること

## Task 2: レポート AMEND（docs/missions/issue9-phase4.1/phase4.1-report.md）
- 結論（§1）: 「A7 を新ベースとして採用」→「**A8 を新ベースとして採用**（10 seed 符号検定・p=.002・事前登録どおりの AMEND）」に更新
- §4 相当の表: 10 seed の error 表 + 符号検定結果を追加（A6 は 3 seed の既存値のまま・「A8 は A6 と同性能 + proper_noun 保持」を明記）
- 事前登録ルール §6 相当: ルール1 の行を「seed 増比較により A8 に AMEND」に更新
- ハーネス制限の注記: summarize_only の SEEDS ハードコードで 10 seed 集計ができず report.json 直接抽出で検定した件（Task 1 で修正済み）
- 書き方規約を厳守: 判別不能は明記・単一 seed p でラベル付けしない・数値は上記正本と一致させる
- 実行環境・artifact SHA-256 の節は更新不要（CE 再実行の結果は後続フェーズで追記される）

## 検証
- `uv run pytest -q` 全件 green（Task 1 の修正で既存テストが壊れないこと）
- `git diff --check`
- レポートの数値が上記正本と完全一致すること

## 制約
- git commit はローカルに（push・PR 作成・issue 操作は禁止）
- 探索コア（src/beme/lattice.py / converter.py）は触らない
- 日本語で報告（変更内容・テスト件数・レポートの更新箇所）
