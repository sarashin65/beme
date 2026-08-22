# Task 5: `lattice.py` の Scorer 配線と状態一般化（issue #9 Phase 1・山場）

`docs/missions/issue9/PLAN.md` のタスク 5 を実装してください。§3.2（ψ の状態制約）、§4.2（累積式の順序・bit-exact の根拠）、§5 タスク5 を読んでから着手すること。

## 実装内容（PLAN.md タスク5 の要点）

1. `k_best_viterbi` / `backward_costs_by_pos` / `backward_viterbi` / `_paths_by_node` の 4 箇所すべてを `scorer` 経由へ変更。**累積式の順序は §4.2-1 のとおり据え置く**（bit-exact を保つため）
2. `backward_costs_by_state(lattice, scorer, *, states=None)` を実装し、既定で「その位置へ入る edge の状態集合」に限定（§3.2）。`backward_costs_by_pos(lattice)` は互換ラッパとして残す
3. `_TRANSITION_COSTS` と `pos_transition_cost` は**削除せず**、`legacy_v1` の重みから導出される薄い互換関数として残す（`__init__.py` の公開 API・既存テストを壊さない）。実体の重複定義は禁止。値の出所は 1 箇所（`legacy_v1` の重み定義）にする
4. `Converter.__init__` に `scorer: Scorer | None = None` を追加（既定 `None` = legacy）
5. `Lattice.k_best` / `paths_at` / `backward_costs` にも `scorer` を透過

## 完了条件（PLAN.md より・すべて確認すること）

- [ ] `tests/test_scorer_golden.py`（タスク1 の golden）が **bit-identical で pass**（cost の `repr` 文字列まで一致）
- [ ] `grep -n "token.cost" src/beme/lattice.py` の結果が tie-break キー（`_edge_key` / `_path_key`）のみ = 探索式に直結の cost 参照が残っていない
- [ ] `scripts/evaluate.py --suite all --reranker mock` の 4 指標が §1.5 と完全一致（CER=0.290080 / exact=0.070000 / Mozc 88/498 / handmade Top-1=1.000000）
- [ ] `pos_transition_cost` の既存テストが無改修で pass
- [ ] `uv run pytest -q` 全通（425 件 + 新規）

## 検証コマンド（PLAN.md より）

```bash
uv run pytest -q
uv run python scripts/evaluate.py --suite all --reranker mock
# 期待: NDCG@5=0.992972 / MRR=1.000000 / Top-1=1.000000 / CER=0.290080 / exact=0.070000 / Mozc 88/498
grep -n "token\.cost" src/beme/lattice.py
uv run python -c "
from beme.dictionary import Dictionary
from beme.lattice import build_lattice, k_best_viterbi
d=Dictionary.from_tsv('data/dictionary.tsv')
p=k_best_viterbi(build_lattice('きしゃのきしゃがきしゃできしゃした',d),beam_width=3)
print([(repr(x.cost),x.surface) for x in p])
"
# 期待: ('5.56','記者の記者が記者で記者した') が先頭
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- **累積式の順序を変えない**（§4.2-1・bit-exact が崩れるため）
- 自己確認した出力（pytest 結果・evaluate 4指標・grep token.cost 結果・K-best 先頭 2 件）を貼って報告すること
