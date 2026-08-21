# Task 3: 結果の集計とレポート作成（k 拡大ゲート実験）

## 指示

`docs/p2-kexp-plan.md` の Task 3 を実施してください。Task 1（--k フラグ対応）・Task 2（k=5/10/20/64 の実験実行）は完了済みで、結果 JSON は worktree 直下にあります（results_k5.json / results_k10.json / results_k20.json / results_k64.json）。

## やってほしいこと

1. 保存した JSON を読み込み、**span 数（single-span / multi-span）ごとの層別集計**を行うスクリプトを作成
   - 各ケースの hypothesis と references から、候補が single-span か multi-span かを判定して集計
   - k=5 / 64 など条件間の出力変化も層別に確認
2. `docs/p2-kexp-report.md` を執筆（プランの §5 構成に従う）:
   - 結論: 実リランカによる改善幅・レイテンシ・P2 への最終判断
   - 指標サマリ: k=5/10/20/64 の比較表
   - span 数別分析
   - 定性分析（正解に昇格したケース・悪化したケース）
   - **「k を広げても出力が 0/200 しか変わらない」理由の考察**（リランカスコアの結合重み・候補分布の観点）

## 完了条件

- 層別集計スクリプトが worktree 内に保存されている
- `docs/p2-kexp-report.md` が完成している（結論が明確に書かれている）

## 制約

- git commit / push はしない
- production コードは変更しない
