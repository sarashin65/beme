# Task 3: 集計・統計判定モジュールと回帰テスト

## 目的

Phase 3 のレビューで見つかった最悪のバグは、McNemar が **A0 の未学習 baseline** と比較していたことだった（未学習 baseline は全条件で同一なので、A0 と 1 件も違わない条件でも `significant_better` が出る）。加えて、判定ラベルが代表 seed 1 本で決まっており、seed の選び方でラベルが反転する状態だった。

**この 2 件はどちらも「集計・検定ロジックが CLI スクリプトの本文に埋まっていて、単体でテストされていなかった」ことが原因である。** Phase 4 は集計・検定を最初からテスト可能なモジュールに切り出し、比較対象・ペアリング・同点処理を契約として固定する。

最初に `docs/missions/issue9-phase4/PLAN.md` の §3.4 と §3.5、`scripts/run_ablations.py`（`mcnemar` は `:61-71`、集計は `summarize_values` 以降）、`scripts/diagnose_cases.py`（`_stratify`）を読むこと。

## 依存

なし。Task 1・2・4 と並行してよい。**ただし Task 2 の `combine` がこのモジュールを使う**ので、関数シグネチャは PLAN §3.5 の表のとおりに固定し、着手前に Task 2 側と食い違わないことを確認すること。

## 対象ファイル

- 新規: `src/beme/rerank_stats.py`
- 新規: `tests/test_rerank_stats.py`

`src/beme/` に置く理由: `scripts/` から import でき、`pytest` から直接テストでき、Task 2 のハーネスと Task 7 のレポート生成の両方が同じ実装を使うため。**`scripts/` 側に集計ロジックを重複させないこと。**

## 実装する関数と契約

### `paired_table(before, after) -> dict`

同一 id で突き合わせ、2×2 分割表 `{a, b, c, d}` を返す（`a` = 両方正解、`b` = before のみ正解 = 失った、`c` = after のみ正解 = 得た、`d` = 両方誤り）。

- **id 集合が一致しなければ例外。** 部分集合でも例外にする。「片方に無い id を黙って落とす」を禁止する。
- 入力の順序に依存しない。
- id の重複があれば例外。
- **`before` と `after` が同一オブジェクトのとき `b = c = 0` になることをテストで固定する**（Phase 3 が検算に使った性質）。

### `mcnemar(b, c) -> dict`

`{"b", "c", "discordant", "method", "p_value"}` を返す。discordant < 25 は両側正確二項、以上は連続修正 χ²。**`scripts/run_ablations.py:61-71` と同一の値**を返すこと（Phase 3 の既知値を golden としてテストで固定する）。

### `label(tables, *, alpha=0.05) -> str`

複数 seed の分割表リストからラベルを決める。

- `significant_better` / `significant_worse` は **記録された全 seed が `p < alpha`** かつ中央値の向きが一致するときだけ。
- それ以外は `indeterminate`。
- **seed が 1 本しか無い場合は `indeterminate` 側へ倒すか、`single_seed` と明示するかを決めてテストで固定する。** 黙って有意ラベルを出さないこと。
- **分割表を seed 間で合算しない**（同じ 2,000 件を 3 回数えると独立性が壊れる）。合算しようとしたら例外にしてよい。

### `top1_from_scores(candidates, *, viterbi_weight, reranker_weight, rerank_k) -> str`

PLAN §3.2 のソートキー。`Converter.convert` と一致すること。**Task 2 のテスト 1 がこの関数を対象にする。**

- ソートキーの第 2・第 3 要素（`cost`・`surface`）が同点時に必ず効くことをテストで固定する。
- `rerank_k` が `None` / 候補数未満 / 候補数以上 の 3 通り。
- `relevance` が `None`（context 空）の候補が 1 つでもあれば、**その例はリランクしない**（並べ替えない）。混在を許さない。

### `stratify(records, axis) -> dict`

軸: `contrast_pair` / `reading_length`（1-3 / 4-6 / 7-10 / 11-16）/ `span_tokens` / `proper_noun`。各層に `n` / `before_error_rate` / `after_error_rate` / `low_support`（n < 30）を持たせる。

### `pair_level(records) -> dict`

`contrast_pair_id` で畳み、`{both_correct, one_correct, both_wrong, incomplete_pairs}` を返す。**対の片側しか split に無い場合の扱いを決めてテストで固定する**（`incomplete_pairs` に数えて `both/one` から除くのが素直）。

### `gold_rank_histogram(records) -> dict`

`{"1": n, "2": n, ..., "20": n, ">20": n, "absent": n}`。gold が候補集合に無い例は `absent`。**`absent` を分母から外さない。**

## 必須テスト

1. `paired_table`: id 欠落・重複・順序入替。**id 集合不一致で例外。**
2. `paired_table`: 同一入力で `b = c = 0`。
3. `mcnemar`: Phase 3 の既知値を golden で固定（例: `b=0, c=30` が `p ≈ 1.2e-7`、`b=c=0` が `p = 1.0`）。discordant = 24 と 25 の境界で method が切り替わる。
4. `label`: 全 seed が p<.05 で同方向 → 有意。1 seed だけ p<.05 → `indeterminate`。中央値の向きが seed 間で割れる → `indeterminate`。
5. `label`: 分割表の合算を試みたら例外（またはそのような API が存在しないこと）。
6. `top1_from_scores`: 同点で `(score, cost, surface)` の順に効く。`rerank_k` 3 通り。`relevance=None` 混在で並べ替えない。
7. `stratify`: 空層・`low_support` フラグ。
8. `pair_level`: 片側だけの対。
9. `gold_rank_histogram`: `absent` が分母に入る。
10. 既存の 585 テストが全通し。

## 完了条件

- 集計・検定が `src/beme/rerank_stats.py` に集約され、`scripts/` に重複していない。
- 上記の契約がすべてテストで固定されている。
- **Phase 3 の 2 つのバグ（比較対象の取り違え・代表 seed 1 本でのラベル決定）が、このモジュールの API では構造的に起こせないこと**を、報告で説明できる。
- `uv run pytest -q` が全通し。

## 検証コマンド

```bash
uv run pytest -q tests/test_rerank_stats.py
uv run pytest -q

# Phase 3 の mcnemar と同値であることの確認
uv run python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from beme.rerank_stats import mcnemar as new
from run_ablations import mcnemar as old
for b,c in ((0,30),(0,0),(6,14),(12,12),(1,24),(1,25),(100,140)):
    a,z = new(b,c), old(b,c)
    assert abs(a['p_value']-z['p_value'])<1e-12 and a['method']==z['method'], (b,c,a,z)
    print(b,c,a['method'],a['p_value'])
print('OK: matches Phase 3')
"
```

## 報告に含めること

- 上記コマンドの**実出力**。
- 「Phase 3 の 2 つのバグがこの API では起こせない理由」の説明（1 段落でよい）。
- `label` が単一 seed をどう扱うことにしたか、その判断。
- `pair_level` が片側だけの対をどう扱うことにしたか、その判断。

## クロスレビュー（必須）

このタスクの完了後、**実装者と別のモデル**で集計・検定部分だけを対象にしたレビューを 1 回入れる。Phase 3 の教訓（ユーザーとの方針確認 5「統計の教訓を最初から織り込む・レビューは実装者と別モデル」）への対応であり、省略しないこと。レビュー結果と対応を報告に含める。

## 共通ルール・制約

- `git commit` / `git push` 禁止。
- 既存の `scripts/run_ablations.py` を書き換えない（Phase 3 の成果物の値が変わる）。新モジュールが同値であることをテストで示すだけでよい。
- ネットワーク・GPU・外部モデルを必要としないテストにする。
