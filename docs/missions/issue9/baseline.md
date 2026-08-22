# Issue #9 Phase 1 Baseline

この baseline は、コミット `4ba97ad` に同梱された `data/dictionary.tsv` を使った
Phase 1 の凍結値である。辞書の SHA-256 は
`efcdf5a1baeaeb74f8ca6c85552d6fac335d40ad274f1e77181c21c8f37f617b`。

```text
Handmade: 14 cases, NDCG@5=0.992972, MRR@5=1.000000, Top-1=1.000000
AJIMEE:   200 cases, mean CER=0.290080, exact=0.070000
Mozc:     88/498 passed (0.176707), 2 skipped
実行時間: 6.028 s（`time` の real）
```

再測定コマンド:

```bash
uv run python scripts/evaluate.py --suite all --reranker mock
```

これは `data/dictionary_learned_v2.tsv` 由来の Issue 記載値（CER `0.274751`、
Mozc `92/498`、handmade Top-1 `0.857143`）とは別の baseline である。後者は
Phase 0 の比較値であり、Phase 1 の再現基準ではない。
