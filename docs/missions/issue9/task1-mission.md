# Task 1: ベースライン凍結（issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 1 を実装してください。§4.3 と §5 タスク1 を読んでから着手すること。

**これはリファクタタスクではありません。`src/beme/lattice.py` を 1 行も変更しないでください。** 後続タスクの検証基準を、現行コードの出力から作るのが目的です。

1. `scripts/dump_kbest.py` を追加する。`--dictionary` / `--k` / `--cases` / `--output` を取り、各読みの K-best を JSON Lines で出力する。float は `json` 既定（`repr` 相当・round-trip する）で書くこと。先頭行に metadata（git rev / 辞書 SHA-256 / k / python version）を入れる。
2. 同スクリプトに `--build-cases` を持たせ、PLAN §4.3-2 の規則（handmade 14 + Mozc 先頭 50 + AJIMEE 先頭 20 のひらがなスパン + 合成ストレス 6）で `tests/data/golden_readings.txt` を決定的に生成し、生成物もコミット対象に含める。
3. `tests/data/kbest_golden_legacy_v1.jsonl` を **現行コードで** K=10 で生成する。
4. `tests/test_scorer_golden.py` を追加し、経路数・各順位の `cost`（文字列表現まで）・`surface`・token 列の一致を assert する。辞書 SHA-256 が不一致なら skip ではなく fail させること。
5. `docs/missions/issue9/baseline.md` に、`uv run python scripts/evaluate.py --suite all --reranker mock` の 4 指標・実行コマンド・`git rev 4ba97ad`・辞書 SHA-256・所要時間を記録する。`data/dictionary_learned_v2.tsv` 由来の issue 記載値（CER 0.274751 等）とは別物である旨を明記すること。

**完了条件**: golden のケース数 ≥ 80 / 全ケースで 1 経路以上 / `tests/test_scorer_golden.py` が現行コードで pass / `uv run pytest -q` 全通（既存 71 件 + 新規）/ `evaluate.py --suite all --reranker mock` が `CER=0.290080・exact=0.070000・Mozc 88/498・handmade Top-1=1.000000` を出す。

**制約**: git commit / push しない。`src/beme/` を変更しない（`git diff --stat src/` が空であることを報告に含める）。自己確認した出力を貼って報告すること。
