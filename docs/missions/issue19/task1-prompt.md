task1（S 実測）を実施してください。

リポジトリ: /home/sarashin/work/beme-19（ブランチ feat/issue19-f2・クリーン）。この worktree の中だけで作業してください。

まず次を順に読んでください:
1. gh issue view 19（本文 + コメント）— 正本
2. docs/missions/issue19/PLAN.md の §1〜§3（前提・4粒度×2候補集合の理由・実測結果が F2 設計をどう分岐させるか）
3. docs/missions/issue19/task1-mission.md — あなたの指示書。目的・出力仕様・完了条件・検証コマンド・制約が全部あります
4. src/beme/features.py / src/beme/scorer.py / src/beme/oracle.py / src/beme/lattice.py / src/beme/training_data.py / scripts/diagnose_cases.py（書き方の作法を真似る）

タスク1 は測定スクリプト 1 本（scripts/measure_local_ceiling.py）+ テスト（tests/test_measure_local_ceiling.py）+ 実測 + レポート（docs/missions/issue19/s-measurement-report.md）です。src/beme/ は 1 行も変更しません。

重要ポイント:
- 主測定は v2 辞書（data/dictionary_v2.tsv）の lattice k-best。JSONL の candidates 列は副測定（v1 辞書由来）
- 4 粒度 (a) lex14 / (b) lex9 / (c) full25 / (d) cost を厳密一致で比較
- 意思決定に効く唯一の指標は「gold と非 gold が同点」の率（gold_vs_nongold_tied_rate）
- 現行 base は A8（decomposed_v1-no_script-surface_len）です
- テストのベースラインは 638 passed / 2 failed。失敗 2 件は tests/test_audit_leakage.py で、data/d1/*.jsonl がこの worktree にあるための環境固有の失敗。直さないでください
- 12,000 ケースの実測は数十分かかり得ます。長引くなら nohup setsid でデタッチ起動して回し、完了を待つ間にレポートを書き進めてください

完了したら検証コマンドを実行し、実際の出力の数値を報告してください。git commit / git push はしないでください。
