# Issue #9 Phase 1 Report

## 結論

Phase 1 の固定重み scorer 基盤と `decomposed_v1` を実装した。基準はリポジトリ同梱の `data/dictionary.tsv` であり、Issue の `dictionary_learned_v2.tsv` baseline とは別物である。

## チェックリストと証跡

| Phase 1 checklist | 状況 | 証跡 |
|---|---|---|
| 現行探索が scorer interface 経由で動く | 充足 | `uv run pytest -q`、`436 passed in 39.09s` |
| 固定重みで現行 1-best を再現 | 充足 | golden 104 cases / K=10 の surface 完全一致 |
| 別プロセスで重みを読み同じ結果を再現 | 充足 | `tests/test_weights.py::test_weights_are_identical_in_a_separate_process` |
| 経路特徴差分を Phase 2 で使える | 充足（抽出基盤） | `path_features` / `feature_delta` とコスト検算テスト |

受け入れ条件のうち正解経路取得、coverage/model error 分離、held-out 比較、handmade 回帰説明は Phase 2/3 の対象外である。

## 指標

| Suite | §1.5 | 再測定（legacy, `dictionary.tsv`） | 再測定（decomposed, `dictionary_v2.tsv`） |
|---|---:|---:|---:|
| Handmade NDCG@5 | 0.992972 | 0.992972 | 0.992972 |
| Handmade MRR@5 | 1.000000 | 1.000000 | 1.000000 |
| Handmade Top-1 | 1.000000 | 1.000000 | 1.000000 |
| AJIMEE mean CER | 0.290080 | 0.290080 | 0.290080 |
| AJIMEE exact | 0.070000 | 0.070000 | 0.070000 |
| Mozc | 88/498 (0.176707) | 88/498 (0.176707) | 88/498 (0.176707) |

実行コマンド:

```text
uv run python scripts/evaluate.py --suite all --reranker mock --weights data/weights/decomposed_v1.json
uv run python scripts/evaluate.py --suite all --reranker mock --dictionary data/dictionary_v2.tsv
```

## Dictionary v2 実測

`data/dictionary_v2.tsv`: 349,043 entries, 31,566,919 bytes。比較対象の v1 は 14,583,715 bytes。実測値は v1 load 3.039 s / v2 load 3.796 s（1.25x）、v1 RSS 176,272 KiB / v2 RSS 207,492 KiB（基準 159,524 KiB 比 1.30x）だった。

```text
uv run python - <<'PY'
from time import perf_counter
from beme.dictionary import Dictionary
started=perf_counter(); d=Dictionary.from_tsv('data/dictionary_v2.tsv')
print(len(d), perf_counter()-started)
PY
# 349043 3.796 s (v1: 3.039 s)
/usr/bin/time -v uv run beme convert きしゃ --dictionary data/dictionary_v2.tsv
# Maximum resident set size: 207492 KiB (v1: 176272 KiB)
```

ゲートは PLAN §5 のロード時間 2.0x 以下、RSS 1.6x 以下。追加列は legacy scorer では参照せず、既存の 4 指標を変更しない。

## dictionary_learned_v2.tsv baseline

`dictionary_learned_v2.tsv` は gitignore された Phase 0 の別 baseline（AJIMEE CER 0.274751、Mozc 92/498、handmade Top-1 0.857143）である。Phase 1 の不変条件・比較基準には使わず、Phase 3 の精度比較へ引き継ぐ。

## Phase 2 引き継ぎ

- `state_key(edge)` を遷移状態の境界として使う。`ψ` は state key のみに依存させる。
- `Φ` は JSON Lines の経路特徴ベクトルで、`path_features` と `feature_delta` を更新式へ接続する。
- 現行 `LatticePath` は位置を持たず Token 列だけを保持する。位置依存特徴や正解経路オラクル DP には、位置を別途扱う設計が必要。
- `decomposed_v1` の初期重みは mapped raw cost / カタカナ / unknown で現行コストを再構成し、他特徴は 0.0。
