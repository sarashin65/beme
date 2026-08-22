# Task 4: `features.py` / `scorer.py` の実装（探索非依存・issue #9 Phase 1）

`docs/missions/issue9/PLAN.md` のタスク 4 を実装してください。§3.4（`legacy_v1` の設計）と §4.1（L1 テスト）を読んでから着手すること。

## 実装内容（PLAN.md タスク4 の要点）

1. `FeatureSpec` / `FeatureExtractor`（§3.4 の `legacy_v1`）を実装
2. `Weights`（この時点ではインメモリのみ・永続化はタスク 6）と `LinearScorer` を実装
3. `default_scorer()` = `legacy_v1` 抽出器 + `legacy_v1` 重み（コード内定数から生成してよい）
4. **`lattice.py` はまだ変更しない**
5. `tests/test_scorer.py`（L1・§4.1）を追加:
   - 19×18 の全組合せで `LinearScorer.transition_cost(prev, edge) == pos_transition_cost(prev, cur)` を `==` で assert
   - `edge_cost(edge) == edge.token.cost` も同様
   - `_TRANSITION_COSTS` の 13 値が `legacy_v1` の重みと 1 対 1 で一致することを assert（写し間違えたら落ちる）

## 完了条件（PLAN.md より）

- [ ] 342 組合せすべてで `==` 一致（許容誤差なし）
- [ ] 特徴抽出が「lexical ちょうど 1 個 / transition ちょうど 1 個」を返すことを assert
- [ ] `lattice.py` の diff が 0 行
- [ ] `uv run pytest -q` 全通（80 件 + 新規）

## 検証

```bash
uv run pytest -q tests/test_scorer.py -v
git diff --stat src/beme/lattice.py   # 空であること
uv run pytest -q
```

## 制約

- git commit / push しない
- 作業はこの worktree（`~/work/beme-9`）内で行う
- `src/beme/lattice.py` は変更しない（`git diff --stat src/beme/lattice.py` が空であることを報告に含める）
- 自己確認した出力（pytest 結果・342 組合せ一致の確認方法と結果）を貼って報告すること
