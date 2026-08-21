# Task 3 続き: docs/p2-kexp-report.md の作成

## 状況

Task 3 の前半（層別集計スクリプト `scripts/analyze_k_expansion.py`）は完了済みで、正常動作を確認しました。

- 集計結果: `analysis_kexp.json`（worktree 直下・k=5/10/20/64 の層別集計済み）
- 実験結果: `results_k5.json` / `results_k10.json` / `results_k20.json` / `results_k64.json`

## やってほしいこと

`docs/p2-kexp-report.md` を完成させてください。`docs/p2-kexp-plan.md` の §5（成果物レポートの構成）に従い:

1. **結論**: 実リランカによる改善幅・レイテンシ・P2（文 K-best 合成再設計）へ進むべきかの最終判断
2. **指標サマリ**: k=5, 10, 20, 64 における CER、Top-1 完全一致率、平均処理時間の比較表
3. **span数別分析**: single-span と multi-span におけるリランカ効果の差異（`analysis_kexp.json` の `span_aggregate` を使用）
4. **定性分析**: リランカによって正解に昇格したケース、および悪化したケースの分析
5. **「k を広げても出力が 0/200 しか変わらない」理由の考察**（リランカスコアの結合重み・候補分布の観点・per_case の具体例 1-2 件を交えて）

## 完了条件

- `docs/p2-kexp-report.md` が完成している（結論が明確に書かれている）

## 制約

- git commit / push はしない
- production コード・既存の結果 JSON は変更しない
- 結果 JSON と analysis_kexp.json の数値から逸脱した数字を書かない
