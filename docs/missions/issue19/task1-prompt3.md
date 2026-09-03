前回の続きです。残作業を完了させてください。

## 現状（観測事実）

- scripts/measure_local_ceiling.py（661行）: 完成・syntax OK
- tests/test_measure_local_ceiling.py: 実装済みだが **5件中1件失敗**
  - `test_required_keys_are_present` が失敗: 手作り JSONL が `training_data.load_jsonl` のバリデーションで `ValueError: candidates and viterbi_costs must have equal lengths`（テストデータの candidates と viterbi_costs の長さ不一致）。他 4 件は pass
- スモークテストは `.smoke/` で実施済み（tie/un の出力あり）

## やること

1. **テスト修正**: `tests/test_measure_local_ceiling.py` の手作り JSONL に `viterbi_costs`（candidates と同数の数値）を追加し、全 5 テストが通るようにする。`training_data.py` の `_parse_example` が要求するフィールド構成（candidates / viterbi_costs / positive 等）をコードで確認して合わせること
2. **スモークテスト**: `uv run python scripts/measure_local_ceiling.py --data data/d1/test.jsonl --limit 50 --output artifacts/issue19/s-measurement/smoke.json`（出力先は repo 内。`/tmp` は権限拒否されるので使わない）
3. **本測定**: 12,000 ケース（train + test）をデタッチ起動で回す
   ```bash
   mkdir -p artifacts/issue19/s-measurement
   nohup setsid uv run python scripts/measure_local_ceiling.py \
     --data data/d1/train.jsonl --data data/d1/test.jsonl \
     --output artifacts/issue19/s-measurement/ties.json \
     > artifacts/issue19/s-measurement/measure.log 2>&1 < /dev/null &
   echo $!   # PID を確認（PPID=1・SID 独立を確認）
   ```
   学習でないので数十分〜1時間程度のはず。完了を待たずに次の作業へ進んでよい
4. **レポート**: `docs/missions/issue19/s-measurement-report.md` を書く（PLAN §3.5 の分岐 A・B 判定・4粒度×2候補集合×3率の表・被覆率表・Opus の辞書レベル 24.9% との比較・SHA-256）
5. **検証**: `uv run pytest -q`（638 + 新規 passed / 2 failed = audit_leakage のみ）・`git diff --stat origin/main -- src/beme/` が空・`git diff --check`

git commit / push はしないでください。完了したら実際の数値を報告してください。
