レポート作成のみのタスクです。スクリプト実装・テスト・測定はすべて完了済みです。

## 現状（完了済み・観測事実）

- scripts/measure_local_ceiling.py（661行）: 完成
- tests/test_measure_local_ceiling.py: 5件実装済み（全 pass）
- pytest: 643 passed / 2 failed（失敗は audit_leakage の環境固有 2 件のみ・直さない）
- 本測定: 12,000 ケース完走
  - artifacts/issue19/s-measurement/ties.json（SHA-256: f5972c811ba86ac977d81fa1d02aa1daa72de148858a3cdff5847eb317bf9ef8）
  - artifacts/issue19/s-measurement/coverage_pairs.jsonl（SHA-256: 6dc370d5e87c77326838f4427cbf657cd4be4b2c860c29401ed4a0f5857a00e2）
  - artifacts/issue19/s-measurement/smoke.json（SHA-256: c063ed05d4dc0880e401e24ecc005ab28a4b6d4556a61c62dd1937bc5805283d）
- feature_distributions も計算済み（例: lex.ipadic_cost_mapped = mean 6.40 / std 2.27 / min 1.75 / max 19.36・count 1,339,192）

## やること

`docs/missions/issue19/s-measurement-report.md` を書くだけです。**コード・テスト・artifact には一切触らないでください。**

ties.json の中身（4粒度 × 2候補集合 × 3率・gold_rank_histogram・misranked_not_tied・feature_distributions・coverage）を読み、以下を含むレポートを書いてください:

1. 先頭に結論と数値（PLAN §3.5 の分岐 A・B のどれに該当したか）
2. 4 粒度 × 2 候補集合 × 3 率の表（train / test）
3. 被覆率の表
4. Opus の辞書レベル 24.9% と d1 候補集合上の lex14 率の比較と説明（辞書 72,013 読み vs d1 8,964 種・分布が異なる等）
5. 分岐 A / B の判定と、それが F2 の期待値に何を意味するか
6. 実行環境（機名・Python version・所要時間）・入力の SHA-256
7. 単一 split の値に p 値ラベルを付けない（Phase 3/4 のレポート規約）

PLAN.md §3.5 の分岐 A（gold_vs_nongold_tied_rate: ≥10% / 3-10% / <3%）と分岐 B（gold 到達不能率: 5% 超なら辞書側が先）を基準に判定してください。

git commit / push はしないでください。完了したらレポートのパスと結論の要約を報告してください。
