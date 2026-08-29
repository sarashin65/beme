# task3 v2 修正の続き: M-D テスト追加のみ

前回の修正（M-A/M-B/M-C）は完了済み・syntax OK です。残っているのは **coverage 経路のテスト追加（M-D）** だけです。これだけを実施してください。

作業ディレクトリ: `/home/sarashin/work/beme-19`（branch `task3-v2`）

## M-D: coverage 経路のテスト追加

`tests/test_corpus_freq.py` に以下を追加してください:

1. **coverage 計算の検証テスト**: 合成 d1（train/test 各 2 行・`span_tokens` 混在）と合成 `coverage_pairs.jsonl` を `tmp_path` に置き、ビルダーを `--articles-jsonl` で実行して、manifest の `d1_test_gold_pair_coverage` / `_span1_only` / `coverage_pairs_match` の期待値を直接 assert する
   - テストは `build_corpus_freq.py` の関数を import して直接呼ぶか、subprocess で実行するかは既存テストの流儀に合わせる
   - `data/d1/*.jsonl` や `data/corpus_freq_v1.bin` に依存しないこと（tmp_path で完結）
2. **coverage 不一致の負テスト**: 合成 d1 と**一致しない** `coverage_pairs.jsonl` を与え、`coverage_pairs_match: false` になり**非ゼロ終了**することを確認（M-C の修正で例外ではなくなったので、終了コードで検証）

既存テスト（9 件）は全部通ったままにすること。

## 検証

```bash
uv run pytest -q tests/test_corpus_freq.py -v   # 11 件（9 + 新規 2）全部 green
uv run pytest -q 2>&1 | tail -3                 # 654 + 2 新規 passed / 2 failed
git diff --check
```

## 制約

- `src/beme/` は変更しない（テストファイルのみ）
- `git commit` / `git push` しない
- 日本語で報告。追加したテスト名と pytest 結果を先頭に
