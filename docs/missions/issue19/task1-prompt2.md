前回の続きです。スクリプト scripts/measure_local_ceiling.py（572行）は完成・syntax OK です。スモークテストの出力先が /tmp だったため権限拒否で中断しました。以下の続きを実施してください。

**注意: スモークテストの出力先を repo 内に変更してください**（/tmp は opencode の非対話モードで権限拒否されます。artifacts/issue19/s-measurement/ や repo 内の任意パスを使うこと）。

1. tests/test_measure_local_ceiling.py を実装する（task1-mission.md §2.4 の 5 項目: 必須キー揃い / 同点構成で any_tied_pair_rate=1.0 / 非同点で 0.0 / gold 不在で gold_unreachable=1 かつ分母除外 / d1 非依存）
2. スモークテスト: `uv run python scripts/measure_local_ceiling.py --data data/d1/test.jsonl --limit 50 --output artifacts/issue19/s-measurement/smoke.json`（出力先を repo 内に）
3. スモークの数値を確認（4 粒度 × 2 候補集合の ties が出力されているか）
4. 12,000 ケースの本測定をデタッチ起動（nohup setsid）で回す
5. docs/missions/issue19/s-measurement-report.md を書く（PLAN §3.5 の分岐 A・B 判定込み）
6. 検証コマンド（pytest・git diff --stat origin/main -- src/beme/ が空・git diff --check）を実行して数値を報告

git commit / push はしないでください。
